import * as THREE from "three"
import {
  clamp,
  dot,
  float,
  mix,
  sin,
  smoothstep,
  type TSLNode,
  uniform,
  vec3,
  vec4,
} from "three/tsl"
import { PassNode } from "@/renderer/pass-node"
import type { LayerParameterValues } from "@/types/editor"

type Node = TSLNode

// Gradient map: luminance → five colour stops (void, shadow, mid, highlight, top), with the
// mid stop drifting toward the shadow colour on a slow sine (the neo-cypherpunk warm→cold cycle).
// Luminance is stretched between black/white point first so dim sources (an ASCII pass) still reach the top.
// Points are linear luminance: 0.45 linear is roughly 0.7 on a display-referred levels tool.

// The chain is linear-light (render targets are half-float, the display pass encodes to sRGB), so the
// picker hex is decoded to linear here. Luminance below is linear luminance too.
function hexToLinearRgb(
  value: unknown,
  fallback: string
): [number, number, number] {
  const hex =
    typeof value === "string" && /^#?[0-9a-fA-F]{6}$/.test(value)
      ? value
      : fallback
  const color = new THREE.Color(hex.startsWith("#") ? hex : `#${hex}`)
  return [color.r, color.g, color.b]
}

function num(
  value: unknown,
  fallback: number,
  min: number,
  max: number
): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(min, Math.min(max, value))
    : fallback
}

export class PaletteMapPass extends PassNode {
  private readonly voidUniform: Node
  private readonly shadowUniform: Node
  private readonly midUniform: Node
  private readonly highlightUniform: Node
  private readonly topUniform: Node
  private readonly blackPointUniform: Node
  private readonly whitePointUniform: Node
  private readonly voidBlendUniform: Node
  private readonly shadowStopUniform: Node
  private readonly midStopUniform: Node
  private readonly highlightStopUniform: Node
  private readonly driftDepthUniform: Node
  private readonly driftRateUniform: Node
  private readonly strengthUniform: Node
  private readonly timeUniform: Node

  constructor(layerId: string) {
    super(layerId)
    this.voidUniform = uniform(new THREE.Vector3(0.039, 0.02, 0.078))
    this.shadowUniform = uniform(new THREE.Vector3(0, 0.157, 0.706))
    this.midUniform = uniform(new THREE.Vector3(1, 0, 0.392))
    this.highlightUniform = uniform(new THREE.Vector3(1, 0.078, 0.576))
    this.topUniform = uniform(new THREE.Vector3(1, 0.706, 0.784))
    this.blackPointUniform = uniform(0.03)
    this.whitePointUniform = uniform(0.45)
    this.voidBlendUniform = uniform(0.12)
    this.shadowStopUniform = uniform(0.25)
    this.midStopUniform = uniform(0.5)
    this.highlightStopUniform = uniform(0.78)
    this.driftDepthUniform = uniform(0.35)
    this.driftRateUniform = uniform((2 * Math.PI) / 6)
    this.strengthUniform = uniform(0.95)
    this.timeUniform = uniform(0)
    this.rebuildEffectNode()
  }

  override updateParams(params: LayerParameterValues): void {
    const setColor = (target: Node, value: unknown, fallback: string) => {
      const [r, g, b] = hexToLinearRgb(value, fallback)
      ;(target as unknown as { value: THREE.Vector3 }).value.set(r, g, b)
    }
    setColor(this.voidUniform, params.voidColor, "#0A0514")
    setColor(this.shadowUniform, params.shadowColor, "#0028B4")
    setColor(this.midUniform, params.midColor, "#FF0064")
    setColor(this.highlightUniform, params.highlightColor, "#FF1493")
    setColor(this.topUniform, params.topColor, "#FFB4C8")

    const black = num(params.blackPoint, 0.03, 0, 1)
    const white = num(params.whitePoint, 0.45, 0, 1)
    this.blackPointUniform.value = Math.min(black, white - 0.001)
    this.whitePointUniform.value = Math.max(white, black + 0.001)
    this.voidBlendUniform.value = num(params.voidBlend, 0.12, 0, 1)

    const shadowStop = num(params.shadowStop, 0.25, 0.01, 0.98)
    const midStop = Math.max(
      shadowStop + 0.01,
      num(params.midStop, 0.5, 0.02, 0.99)
    )
    const highlightStop = Math.max(
      midStop + 0.01,
      num(params.highlightStop, 0.78, 0.03, 1)
    )
    this.shadowStopUniform.value = shadowStop
    this.midStopUniform.value = midStop
    this.highlightStopUniform.value = highlightStop

    this.driftDepthUniform.value = num(params.driftDepth, 0.35, 0, 1)
    const period = num(params.driftPeriod, 6, 0.5, 60)
    this.driftRateUniform.value = (2 * Math.PI) / period
    this.strengthUniform.value = num(params.strength, 0.95, 0, 1)
  }

  protected override beforeRender(time: number, _delta: number): void {
    this.timeUniform.value = time
  }

  protected override buildEffectNode(): Node {
    if (!this.strengthUniform) {
      return this.inputNode
    }

    const source = vec3(
      float(this.inputNode.r),
      float(this.inputNode.g),
      float(this.inputNode.b)
    )
    const rawLuma = dot(source, vec3(0.299, 0.587, 0.114))
    const luma = smoothstep(
      this.blackPointUniform,
      this.whitePointUniform,
      rawLuma
    )

    const osc = sin(this.timeUniform.mul(this.driftRateUniform))
      .mul(0.5)
      .add(0.5)
    const voidColor = mix(
      this.voidUniform,
      this.shadowUniform,
      this.voidBlendUniform
    )
    const midColor = mix(
      this.midUniform,
      this.shadowUniform,
      osc.mul(this.driftDepthUniform)
    )

    const s1 = mix(
      voidColor,
      this.shadowUniform,
      smoothstep(float(0), this.shadowStopUniform, luma)
    )
    const s2 = mix(
      s1,
      midColor,
      smoothstep(this.shadowStopUniform, this.midStopUniform, luma)
    )
    const s3 = mix(
      s2,
      this.highlightUniform,
      smoothstep(this.midStopUniform, this.highlightStopUniform, luma)
    )
    const mapped = mix(
      s3,
      this.topUniform,
      smoothstep(this.highlightStopUniform, float(1), luma)
    )

    const out = clamp(
      mix(source, mapped, this.strengthUniform),
      vec3(0, 0, 0),
      vec3(1, 1, 1)
    )
    return vec4(out, float(this.inputNode.a))
  }
}
