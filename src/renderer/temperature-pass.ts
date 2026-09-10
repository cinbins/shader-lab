import {
  abs,
  atan,
  clamp,
  cos,
  float,
  max,
  mix,
  pow,
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

const cbrt = (x: Node) => pow(max(x, float(1e-6)), float(1 / 3))

function linearToOklab(c: Node): Node {
  const l = cbrt(
    c.r.mul(0.4122214708).add(c.g.mul(0.5363325363)).add(c.b.mul(0.0514459929))
  )
  const m = cbrt(
    c.r.mul(0.2119034982).add(c.g.mul(0.6806995451)).add(c.b.mul(0.1073969566))
  )
  const s = cbrt(
    c.r.mul(0.0883024619).add(c.g.mul(0.2817188376)).add(c.b.mul(0.6299787005))
  )
  return vec3(
    l.mul(0.2104542553).add(m.mul(0.793617785)).sub(s.mul(0.0040720468)),
    l.mul(1.9779984951).sub(m.mul(2.428592205)).add(s.mul(0.4505937099)),
    l.mul(0.0259040371).add(m.mul(0.7827717662)).sub(s.mul(0.808675766))
  )
}

function oklabToLinear(lab: Node): Node {
  const l_ = lab.x.add(lab.y.mul(0.3963377774)).add(lab.z.mul(0.2158037573))
  const m_ = lab.x.sub(lab.y.mul(0.1055613458)).sub(lab.z.mul(0.0638541728))
  const s_ = lab.x.sub(lab.y.mul(0.0894841775)).sub(lab.z.mul(1.291485548))
  const l = l_.mul(l_).mul(l_)
  const m = m_.mul(m_).mul(m_)
  const s = s_.mul(s_).mul(s_)
  return vec3(
    l.mul(4.0767416621).sub(m.mul(3.3077115913)).add(s.mul(0.2309699292)),
    l.mul(-1.2684380046).add(m.mul(2.6097574011)).sub(s.mul(0.3413193965)),
    l.mul(-0.0041960863).sub(m.mul(0.7034186147)).add(s.mul(1.707614701))
  )
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
