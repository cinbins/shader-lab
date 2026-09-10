import * as THREE from "three"
import {
  clamp,
  dot,
  float,
  mix,
  smoothstep,
  type TSLNode,
  uniform,
  vec3,
  vec4,
} from "three/tsl"
import { PassNode } from "@/renderer/pass-node"
import type { LayerParameterValues } from "@/types/editor"

type Node = TSLNode

// Gradient map: luminance → five colour stops (void, shadow, mid, highlight, top). Shadow/mid/highlight are
// each interpolated between a warm pole (red/magenta) and a cold pole (blue/cyan) by one `temperature`
// parameter (0 → 1), on the CPU in OKLab so the midpoint stays a clean violet; keyframes and audio links
// drive `temperature` through updateParams. Void and top are constant (black void keys out under Screen).
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

// OKLab (Björn Ottosson) on linear sRGB, CPU side.
type Rgb = [number, number, number]
function linearToOklab([r, g, b]: Rgb): Rgb {
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}
function oklabToLinear([L, a, b]: Rgb): Rgb {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
  const clamp01 = (v: number) => Math.max(0, Math.min(1, v))
  return [
    clamp01(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    clamp01(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    clamp01(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s),
  ]
}
function mixOklab(warm: Rgb, cold: Rgb, t: number): Rgb {
  const a = linearToOklab(warm)
  const b = linearToOklab(cold)
  return oklabToLinear([
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ])
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
  private readonly shadowStopUniform: Node
  private readonly midStopUniform: Node
  private readonly highlightStopUniform: Node
  private readonly strengthUniform: Node

  constructor(layerId: string) {
    super(layerId)
    this.voidUniform = uniform(new THREE.Vector3(0, 0, 0))
    this.shadowUniform = uniform(new THREE.Vector3(0.155, 0, 0.031))
    this.midUniform = uniform(new THREE.Vector3(1, 0, 0.128))
    this.highlightUniform = uniform(new THREE.Vector3(1, 0.008, 0.291))
    this.topUniform = uniform(new THREE.Vector3(1, 0.451, 0.578))
    this.blackPointUniform = uniform(0.02)
    this.whitePointUniform = uniform(0.55)
    this.shadowStopUniform = uniform(0.22)
    this.midStopUniform = uniform(0.5)
    this.highlightStopUniform = uniform(0.8)
    this.strengthUniform = uniform(0.9)
    this.rebuildEffectNode()
  }

  override updateParams(params: LayerParameterValues): void {
    const setColor = (target: Node, value: unknown, fallback: string) => {
      const [r, g, b] = hexToLinearRgb(value, fallback)
      ;(target as unknown as { value: THREE.Vector3 }).value.set(r, g, b)
    }
    setColor(this.voidUniform, params.voidColor, "#000000")
    setColor(this.topUniform, params.topColor, "#FFB4C8")
    const t = num(params.temperature, 0, 0, 1)
    const setPole = (
      target: Node,
      warm: unknown,
      wf: string,
      cold: unknown,
      cf: string
    ) => {
      const [r, g, b] = mixOklab(
        hexToLinearRgb(warm, wf),
        hexToLinearRgb(cold, cf),
        t
      )
      ;(target as unknown as { value: THREE.Vector3 }).value.set(r, g, b)
    }
    setPole(
      this.shadowUniform,
      params.shadowColor,
      "#6E0032",
      params.coldShadowColor,
      "#0028B4"
    )
    setPole(
      this.midUniform,
      params.midColor,
      "#FF0064",
      params.coldMidColor,
      "#2A6BFF"
    )
    setPole(
      this.highlightUniform,
      params.highlightColor,
      "#FF1493",
      params.coldHighlightColor,
      "#00DCFF"
    )

    const black = num(params.blackPoint, 0.02, 0, 1)
    const white = num(params.whitePoint, 0.55, 0, 1)
    this.blackPointUniform.value = Math.min(black, white - 0.001)
    this.whitePointUniform.value = Math.max(white, black + 0.001)

    const shadowStop = num(params.shadowStop, 0.22, 0.01, 0.98)
    const midStop = Math.max(
      shadowStop + 0.01,
      num(params.midStop, 0.5, 0.02, 0.99)
    )
    const highlightStop = Math.max(
      midStop + 0.01,
      num(params.highlightStop, 0.8, 0.03, 1)
    )
    this.shadowStopUniform.value = shadowStop
    this.midStopUniform.value = midStop
    this.highlightStopUniform.value = highlightStop
    this.strengthUniform.value = num(params.strength, 0.9, 0, 1)
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

    const s1 = mix(
      this.voidUniform,
      this.shadowUniform,
      smoothstep(float(0), this.shadowStopUniform, luma)
    )
    const s2 = mix(
      s1,
      this.midUniform,
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
