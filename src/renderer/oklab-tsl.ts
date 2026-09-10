import { float, max, pow, type TSLNode, vec3 } from "three/tsl"

type Node = TSLNode

// OKLab (Björn Ottosson) on linear sRGB, TSL side. Inputs are expected in [0, 1].
const cbrt = (x: Node) => pow(max(x, float(1e-6)), float(1 / 3))

export function linearToOklab(c: Node): Node {
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

export function oklabToLinear(lab: Node): Node {
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
