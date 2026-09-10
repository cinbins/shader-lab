import {
  abs,
  atan,
  clamp,
  cos,
  float,
  mix,
  sin,
  smoothstep,
  type TSLNode,
  uniform,
  vec3,
  vec4,
} from "three/tsl"
import { linearToOklab, oklabToLinear } from "@/renderer/oklab-tsl"
import { PassNode } from "@/renderer/pass-node"
import type { LayerParameterValues } from "@/types/editor"

type Node = TSLNode

// Temperature: the emotion override. Rotates every colour's hue in OKLab toward a warm pole (red/magenta)
// or a cold pole (blue/cyan) while keeping lightness and chroma, so white highlights and black stay put.
// temperature −1 = fully warm, 0 = untouched, +1 = fully cold. Sits above the Palette Map in the house stack.

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

export class TemperaturePass extends PassNode {
  private readonly temperatureUniform: Node
  private readonly amountUniform: Node
  private readonly warmHueUniform: Node
  private readonly coldHueUniform: Node
  private readonly chromaBoostUniform: Node

  constructor(layerId: string) {
    super(layerId)
    this.temperatureUniform = uniform(0)
    this.amountUniform = uniform(1)
    this.warmHueUniform = uniform((350 * Math.PI) / 180)
    this.coldHueUniform = uniform((250 * Math.PI) / 180)
    this.chromaBoostUniform = uniform(0)
    this.rebuildEffectNode()
  }

  override updateParams(params: LayerParameterValues): void {
    this.temperatureUniform.value = num(params.temperature, 0, -1, 1)
    this.amountUniform.value = num(params.amount, 1, 0, 1)
    this.warmHueUniform.value =
      (num(params.warmHue, 350, 0, 360) * Math.PI) / 180
    this.coldHueUniform.value =
      (num(params.coldHue, 250, 0, 360) * Math.PI) / 180
    this.chromaBoostUniform.value = num(params.chromaBoost, 0, 0, 1)
  }

  protected override buildEffectNode(): Node {
    if (!this.temperatureUniform) {
      return this.inputNode
    }
    const source = vec3(
      float(this.inputNode.r),
      float(this.inputNode.g),
      float(this.inputNode.b)
    )
    const lab = linearToOklab(clamp(source, vec3(0, 0, 0), vec3(1, 1, 1)))
    const L = lab.x
    const chroma = lab.y.mul(lab.y).add(lab.z.mul(lab.z)).sqrt()
    const hue = atan(lab.z, lab.y)
    const t = this.temperatureUniform
    const target = mix(
      this.warmHueUniform,
      this.coldHueUniform,
      smoothstep(float(-1), float(1), t)
    )
    // shortest-path hue interpolation
    const TWO_PI = float(Math.PI * 2)
    let delta: Node = target.sub(hue)
    delta = delta.sub(TWO_PI.mul(delta.div(TWO_PI).add(0.5).floor()))
    // neutrals (low chroma) are left alone so whites/blacks never tint
    const weight = abs(t)
      .mul(this.amountUniform)
      .mul(smoothstep(float(0.005), float(0.03), chroma))
    const newHue = hue.add(delta.mul(weight))
    const newChroma = chroma.mul(
      float(1).add(this.chromaBoostUniform.mul(abs(t)))
    )
    const out = oklabToLinear(
      vec3(L, newChroma.mul(cos(newHue)), newChroma.mul(sin(newHue)))
    )
    return vec4(
      clamp(out, vec3(0, 0, 0), vec3(1, 1, 1)),
      float(this.inputNode.a)
    )
  }
}
