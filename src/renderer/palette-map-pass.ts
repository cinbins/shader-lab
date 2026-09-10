import * as THREE from "three"
import {
  clamp,
  dot,
  float,
  mix,
  pow,
  smoothstep,
  type TSLNode,
  uniform,
  vec3,
  vec4,
} from "three/tsl"
import { PassNode } from "@/renderer/pass-node"
import type { LayerParameterValues } from "@/types/editor"

type Node = TSLNode

// Gradient map: luminance → a fixed nine-stop colour ramp (the ATB neo-cypherpunk house ramp, sampled from
// DQ's reference: black → deep indigo → cobalt → electric → violet → magenta → rose → red → top).
// Stops are evenly spaced after a gamma curve; black/white point stretch the source first so dim sources
// (an ASCII pass) reach the top. Void stays pure black so the pass keys out under Screen in the NLE.
// Emotion (warm/cold) is NOT here: it is the separate Temperature layer above this one.
// The chain is linear-light (half-float targets, sRGB encode on display): picker hex is decoded to linear.

export const PALETTE_STOPS = [
  ["voidColor", "#000000", "Void"],
  ["stop1Color", "#12103A", "Deep"],
  ["stop2Color", "#2E22A3", "Cobalt"],
  ["stop3Color", "#4723AB", "Electric"],
  ["stop4Color", "#7925B0", "Violet"],
  ["stop5Color", "#952397", "Magenta"],
  ["stop6Color", "#BE1F5B", "Rose"],
  ["stop7Color", "#CB2028", "Red"],
  ["topColor", "#FFB4C8", "Top"],
] as const

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
  private readonly stopUniforms: Node[]
  private readonly blackPointUniform: Node
  private readonly whitePointUniform: Node
  private readonly gammaUniform: Node
  private readonly strengthUniform: Node

  constructor(layerId: string) {
    super(layerId)
    this.stopUniforms = PALETTE_STOPS.map(([, hex]) => {
      const [r, g, b] = hexToLinearRgb(hex, hex)
      return uniform(new THREE.Vector3(r, g, b))
    })
    this.blackPointUniform = uniform(0.02)
    this.whitePointUniform = uniform(0.55)
    this.gammaUniform = uniform(1)
    this.strengthUniform = uniform(0.9)
    this.rebuildEffectNode()
  }

  override updateParams(params: LayerParameterValues): void {
    PALETTE_STOPS.forEach(([key, fallback], i) => {
      const [r, g, b] = hexToLinearRgb(params[key], fallback)
      ;(this.stopUniforms[i] as unknown as { value: THREE.Vector3 }).value.set(
        r,
        g,
        b
      )
    })
    const black = num(params.blackPoint, 0.02, 0, 1)
    const white = num(params.whitePoint, 0.55, 0, 1)
    this.blackPointUniform.value = Math.min(black, white - 0.001)
    this.whitePointUniform.value = Math.max(white, black + 0.001)
    this.gammaUniform.value = num(params.gamma, 1, 0.3, 3)
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
    const luma = pow(
      smoothstep(this.blackPointUniform, this.whitePointUniform, rawLuma),
      this.gammaUniform
    )
    const stops = this.stopUniforms
    const n = stops.length - 1
    let mapped: Node = stops[0] as Node
    for (let i = 1; i <= n; i++) {
      mapped = mix(
        mapped,
        stops[i] as Node,
        smoothstep(float((i - 1) / n), float(i / n), luma)
      )
    }
    const out = clamp(
      mix(source, mapped, this.strengthUniform),
      vec3(0, 0, 0),
      vec3(1, 1, 1)
    )
    return vec4(out, float(this.inputNode.a))
  }
}
