import {
  abs,
  clamp,
  dot,
  Fn,
  float,
  floor,
  fract,
  Loop,
  max,
  min,
  mix,
  pow,
  sin,
  smoothstep,
  sqrt,
  step,
  type TSLNode,
  texture as tslTexture,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from "three/tsl"
import * as THREE from "three/webgpu"
import { getCompositionFrame } from "@/lib/editor/composition"
import {
  ASCII_CHARSETS,
  ASCII_FEATURE_TEXELS,
  type AsciiAtlas,
  type AsciiAtlasOptions,
  type AsciiTonemap,
  buildAsciiAtlas,
  buildRampLutTexture,
  DEFAULT_ASCII_CHARS,
  DEFAULT_EDGE_CHARS,
  isAsciiFontReady,
  loadAsciiFont,
  RAMP_LUT_SIZE,
} from "@/renderer/ascii-atlas"
import { GridRenderPass } from "@/renderer/grid-render-pass"
import { PassNode } from "@/renderer/pass-node"
import type { LayerParameterValues, SceneConfig } from "@/types/editor"

type Node = TSLNode
type AsciiColorMode = "monochrome" | "source"
type AsciiCharset = keyof typeof ASCII_CHARSETS | "custom"

const ATLAS_INNER_HEIGHT = 64
const DEFAULT_FONT_FAMILY = "mono"
const SUPERSAMPLE = 3
const MAX_GRID_DIMENSION = 4096
const DETAIL = 4
const DETAIL_DIM = DETAIL * DETAIL
const STRUCTURE_CANDIDATES = 12
const STRUCTURE_WINDOW = 0.1
const SHUFFLE_WINDOW = 0.05
const INDEX_HI_SCALE = 256

const BREAK_LEVELS: Record<string, number> = {
  "2x": 1,
  "4x": 2,
  "8x": 3,
  "16x": 4,
  off: 0,
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function parseCssColorRgb(value: string): [number, number, number] {
  const rgba = value.match(
    /rgba?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*[\d.]+)?\s*\)/i
  )

  if (rgba) {
    const color = new THREE.Color().setRGB(
      clamp01(Number.parseFloat(rgba[1] ?? "0") / 255),
      clamp01(Number.parseFloat(rgba[2] ?? "0") / 255),
      clamp01(Number.parseFloat(rgba[3] ?? "0") / 255),
      THREE.SRGBColorSpace
    )

    return [color.r, color.g, color.b]
  }

  const hex = value.trim().replace("#", "")

  if (hex.length === 6 || hex.length === 3) {
    const color = new THREE.Color(`#${hex}`)

    return [color.r, color.g, color.b]
  }

  return [1, 1, 1]
}

export class AsciiPass extends PassNode {
  private atlas: AsciiAtlas | null = null
  private lutTexture: THREE.DataTexture | null = null
  private atlasTextureNodes: Node[] = []
  private featureTextureNodes: Node[] = []
  private lutTextureNodes: Node[] = []
  private analysisSourceNodes: Node[] = []
  private detailSourceNodes: Node[] = []
  private retiredAtlasTextures: THREE.Texture[] = []
  private framesSinceAtlasSwap = 0
  private fontLoadToken = 0
  private logicalWidth = 1
  private logicalHeight = 1
  private outputWidth = 1
  private gridWidth = 1
  private gridHeight = 1
  private rebuildGeneration = 0

  private readonly analysisPass: GridRenderPass
  private readonly detailPass: GridRenderPass
  private readonly layoutPass: GridRenderPass
  private readonly selectPass: GridRenderPass

  private readonly atlasColumnsUniform: Node
  private readonly atlasInnerHeightUniform: Node
  private readonly atlasInnerXUniform: Node
  private readonly atlasInnerYUniform: Node
  private readonly atlasPadXUniform: Node
  private readonly atlasPadYUniform: Node
  private readonly atlasRowsUniform: Node
  private readonly bgOpacityUniform: Node
  private readonly boldnessUniform: Node
  private readonly breakThresholdUniform: Node
  private readonly cellAspectUniform: Node
  private readonly cellPixelHeightUniform: Node
  private readonly cellUvWidthUniform: Node
  private readonly cellUvHeightUniform: Node
  private readonly gridOriginXUniform: Node
  private readonly gridOriginYUniform: Node
  private readonly gridHeightUniform: Node
  private readonly gridWidthUniform: Node
  private readonly invertUniform: Node
  private readonly monoBlueUniform: Node
  private readonly monoGreenUniform: Node
  private readonly monoRedUniform: Node
  private readonly placeholder: THREE.Texture
  private readonly rampCountUniform: Node
  private readonly renderScaleUniform: Node
  private readonly rowWarpUniform: Node
  private readonly sdfRadiusUniform: Node
  private readonly shuffleUniform: Node
  private readonly shuffleSeedUniform: Node
  private readonly signalBlackPointUniform: Node
  private readonly signalWhitePointUniform: Node
  private readonly sourceMixUniform: Node
  private readonly structureUniform: Node

  private sceneConfig: SceneConfig | null = null
  private currentColumns = 80
  private currentBreakLevels = 0
  private currentRowWarpEnabled = false
  private currentStructureEnabled = false
  private currentShuffleRate = 0
  private currentTonemap: AsciiTonemap = "rank"
  private currentCharset: AsciiCharset = "light"
  private currentCustomChars = DEFAULT_ASCII_CHARS
  private currentFontFamily = DEFAULT_FONT_FAMILY
  private currentFontWeight = 400
  private currentColorMode: AsciiColorMode = "monochrome"

  constructor(layerId: string) {
    super(layerId)
    this.placeholder = new THREE.Texture()
    this.analysisPass = new GridRenderPass({ linear: true })
    this.detailPass = new GridRenderPass({ linear: true })
    this.layoutPass = new GridRenderPass()
    this.selectPass = new GridRenderPass()
    this.atlasColumnsUniform = uniform(1)
    this.atlasInnerHeightUniform = uniform(ATLAS_INNER_HEIGHT)
    this.atlasInnerXUniform = uniform(1)
    this.atlasInnerYUniform = uniform(1)
    this.atlasPadXUniform = uniform(0)
    this.atlasPadYUniform = uniform(0)
    this.atlasRowsUniform = uniform(1)
    this.bgOpacityUniform = uniform(0)
    this.boldnessUniform = uniform(0)
    this.breakThresholdUniform = uniform(0.06)
    this.cellAspectUniform = uniform(0.6)
    this.cellPixelHeightUniform = uniform(12)
    this.cellUvWidthUniform = uniform(0.01)
    this.cellUvHeightUniform = uniform(0.01)
    this.gridOriginXUniform = uniform(0)
    this.gridOriginYUniform = uniform(0)
    this.gridHeightUniform = uniform(1)
    this.gridWidthUniform = uniform(1)
    this.invertUniform = uniform(0)
    this.monoBlueUniform = uniform(0.94)
    this.monoGreenUniform = uniform(0.96)
    this.monoRedUniform = uniform(0.96)
    this.rampCountUniform = uniform(DEFAULT_ASCII_CHARS.length)
    this.renderScaleUniform = uniform(1)
    this.rowWarpUniform = uniform(0)
    this.sdfRadiusUniform = uniform(8)
    this.shuffleUniform = uniform(0)
    this.shuffleSeedUniform = uniform(0)
    this.signalBlackPointUniform = uniform(0)
    this.signalWhitePointUniform = uniform(1)
    this.sourceMixUniform = uniform(0)
    this.structureUniform = uniform(0)
    this.rebuildAtlas()
    this.analysisPass.setColorNode(this.buildAnalysisColorNode())
    this.detailPass.setColorNode(this.buildDetailColorNode())
    this.layoutPass.setColorNode(this.buildLayoutColorNode())
    this.selectPass.setColorNode(this.buildSelectColorNode())
    this.rebuildEffectNode()
  }

  override render(
    renderer: THREE.WebGPURenderer,
    inputTexture: THREE.Texture,
    outputTarget: THREE.WebGLRenderTarget,
    time: number,
    delta: number
  ): void {
    this.syncGridSize()
    for (const node of this.analysisSourceNodes) {
      node.value = inputTexture
    }
    for (const node of this.detailSourceNodes) {
      node.value = inputTexture
    }

    const atlasTexture = this.atlas?.texture

    if (atlasTexture) {
      for (const node of this.atlasTextureNodes) {
        node.value = atlasTexture
      }
    }

    const featureTexture = this.atlas?.featureTexture

    if (featureTexture) {
      for (const node of this.featureTextureNodes) {
        node.value = featureTexture
      }
    }

    if (this.lutTexture) {
      for (const node of this.lutTextureNodes) {
        node.value = this.lutTexture
      }
    }

    this.shuffleSeedUniform.value =
      this.currentShuffleRate > 0
        ? Math.floor(time * this.currentShuffleRate)
        : 0

    this.analysisPass.render(renderer)

    if (this.currentStructureEnabled) {
      this.detailPass.render(renderer)
    }

    if (this.currentBreakLevels > 0) {
      this.layoutPass.render(renderer)
    }

    this.selectPass.render(renderer)

    super.render(renderer, inputTexture, outputTarget, time, delta)

    if (this.retiredAtlasTextures.length > 0) {
      this.framesSinceAtlasSwap += 1

      if (this.framesSinceAtlasSwap >= 3) {
        for (const texture of this.retiredAtlasTextures) {
          texture.dispose()
        }
        this.retiredAtlasTextures = []
      }
    }
  }

  override updateParams(params: LayerParameterValues): void {
    this.currentColumns =
      typeof params.columns === "number"
        ? Math.max(4, Math.min(400, Math.round(params.columns)))
        : 80

    const nextCharset = this.resolveCharset(params.charset)
    const nextCustomChars =
      typeof params.customChars === "string"
        ? params.customChars
        : DEFAULT_ASCII_CHARS
    const nextFontFamily =
      typeof params.fontFamily === "string" && params.fontFamily.length > 0
        ? params.fontFamily
        : DEFAULT_FONT_FAMILY
    const nextFontWeight =
      typeof params.fontWeight === "number"
        ? Math.max(100, Math.min(900, Math.round(params.fontWeight)))
        : 400
    const nextTonemap: AsciiTonemap =
      params.tonemap === "coverage" ? "coverage" : "rank"

    const needsAtlasRebuild =
      nextCharset !== this.currentCharset ||
      nextFontFamily !== this.currentFontFamily ||
      nextFontWeight !== this.currentFontWeight ||
      (nextCharset === "custom" && nextCustomChars !== this.currentCustomChars)
    const needsLutRebuild = nextTonemap !== this.currentTonemap

    this.currentCharset = nextCharset
    this.currentCustomChars = nextCustomChars
    this.currentFontFamily = nextFontFamily
    this.currentFontWeight = nextFontWeight
    this.currentTonemap = nextTonemap

    if (needsAtlasRebuild) {
      this.rebuildAtlas()
    } else if (needsLutRebuild) {
      this.rebuildLut()
    }

    this.boldnessUniform.value =
      typeof params.boldness === "number"
        ? Math.max(-1, Math.min(1, params.boldness))
        : 0
    this.bgOpacityUniform.value =
      typeof params.bgOpacity === "number" ? clamp01(params.bgOpacity) : 0
    this.breakThresholdUniform.value =
      typeof params.breakThreshold === "number"
        ? Math.max(0.001, Math.min(0.5, params.breakThreshold))
        : 0.06
    this.signalBlackPointUniform.value =
      typeof params.signalBlackPoint === "number"
        ? clamp01(params.signalBlackPoint)
        : 0
    this.signalWhitePointUniform.value =
      typeof params.signalWhitePoint === "number"
        ? clamp01(params.signalWhitePoint)
        : 1
    this.invertUniform.value = params.invert === true ? 1 : 0

    this.currentColorMode =
      params.colorMode === "source" ? "source" : "monochrome"
    this.sourceMixUniform.value = this.currentColorMode === "source" ? 1 : 0

    const [red, green, blue] = parseCssColorRgb(
      typeof params.monoColor === "string" ? params.monoColor : "#f5f5f0"
    )
    this.monoRedUniform.value = red
    this.monoGreenUniform.value = green
    this.monoBlueUniform.value = blue

    const nextShuffle =
      typeof params.shuffle === "number" ? clamp01(params.shuffle) : 0
    this.shuffleUniform.value = nextShuffle
    this.currentShuffleRate =
      typeof params.shuffleRate === "number"
        ? Math.max(0, Math.min(60, params.shuffleRate))
        : 0

    const nextStructure =
      typeof params.structure === "number" ? clamp01(params.structure) : 0
    this.structureUniform.value = nextStructure
    const nextStructureEnabled = nextStructure > 0

    const nextRowWarp =
      typeof params.rowWarp === "number" ? clamp01(params.rowWarp) : 0
    this.rowWarpUniform.value = nextRowWarp
    const nextRowWarpEnabled = nextRowWarp > 0

    const nextBreakLevels =
      BREAK_LEVELS[
        typeof params.breakGrid === "string" ? params.breakGrid : "off"
      ] ?? 0

    const structuralChange =
      nextBreakLevels !== this.currentBreakLevels ||
      nextRowWarpEnabled !== this.currentRowWarpEnabled ||
      nextStructureEnabled !== this.currentStructureEnabled

    this.currentBreakLevels = nextBreakLevels
    this.currentRowWarpEnabled = nextRowWarpEnabled
    this.currentStructureEnabled = nextStructureEnabled

    if (structuralChange) {
      this.scheduleStructuralRebuild()
    }
  }

  override resize(width: number, _height: number): void {
    this.outputWidth = Math.max(1, width)
    this.recomputeRenderScale()
  }

  override updateLogicalSize(width: number, height: number): void {
    this.logicalWidth = Math.max(1, width)
    this.logicalHeight = Math.max(1, height)
    this.recomputeRenderScale()
  }

  override updateSceneConfig(config: SceneConfig): boolean {
    this.sceneConfig = config
    return false
  }

  override dispose(): void {
    this.placeholder.dispose()
    this.atlas?.texture.dispose()
    this.atlas?.featureTexture.dispose()
    this.lutTexture?.dispose()
    for (const texture of this.retiredAtlasTextures) {
      texture.dispose()
    }
    this.retiredAtlasTextures = []
    this.analysisPass.dispose()
    this.detailPass.dispose()
    this.layoutPass.dispose()
    this.selectPass.dispose()
    super.dispose()
  }

  private recomputeRenderScale(): void {
    this.renderScaleUniform.value = Math.max(
      0.1,
      this.outputWidth / Math.max(1, this.logicalWidth)
    )
  }

  private getCompositionFrameSize(): {
    height: number
    width: number
    x: number
    y: number
  } {
    if (!this.sceneConfig) {
      return {
        height: this.logicalHeight,
        width: this.logicalWidth,
        x: 0,
        y: 0,
      }
    }

    return getCompositionFrame(this.sceneConfig, {
      height: this.logicalHeight,
      width: this.logicalWidth,
    })
  }

  private syncGridSize(): void {
    const aspect = this.atlas?.cellAspect ?? 0.6
    const frame = this.getCompositionFrameSize()

    const cellWidth = Math.max(
      0.5,
      frame.width / Math.max(1, this.currentColumns)
    )
    const cellHeight = Math.max(1, cellWidth / aspect)

    const cellUvWidth = cellWidth / this.logicalWidth
    const cellUvHeight = cellHeight / this.logicalHeight

    const frameXUv = frame.x / this.logicalWidth
    const frameYUv = frame.y / this.logicalHeight
    const originX = frameXUv - Math.ceil(frameXUv / cellUvWidth) * cellUvWidth
    const originY = frameYUv - Math.ceil(frameYUv / cellUvHeight) * cellUvHeight

    const gridWidth = Math.min(
      MAX_GRID_DIMENSION,
      Math.max(1, Math.ceil((1 - originX) / cellUvWidth))
    )
    const gridHeight = Math.min(
      MAX_GRID_DIMENSION,
      Math.max(1, Math.ceil((1 - originY) / cellUvHeight))
    )

    this.cellPixelHeightUniform.value = cellHeight
    this.cellUvWidthUniform.value = cellUvWidth
    this.cellUvHeightUniform.value = cellUvHeight
    this.gridOriginXUniform.value = originX
    this.gridOriginYUniform.value = originY

    if (gridWidth === this.gridWidth && gridHeight === this.gridHeight) {
      return
    }

    this.gridWidth = gridWidth
    this.gridHeight = gridHeight
    this.gridWidthUniform.value = gridWidth
    this.gridHeightUniform.value = gridHeight
    this.analysisPass.setSize(gridWidth, gridHeight)
    this.detailPass.setSize(
      Math.min(MAX_GRID_DIMENSION, gridWidth * DETAIL),
      Math.min(MAX_GRID_DIMENSION, gridHeight * DETAIL)
    )
    this.layoutPass.setSize(gridWidth, gridHeight)
    this.selectPass.setSize(gridWidth, gridHeight)
  }

  private getCellUvSize(): Node {
    return vec2(this.cellUvWidthUniform, this.cellUvHeightUniform)
  }

  private getGridOrigin(): Node {
    return vec2(this.gridOriginXUniform, this.gridOriginYUniform)
  }

  private getGridSize(): Node {
    return vec2(this.gridWidthUniform, this.gridHeightUniform)
  }

  private trackAnalysisSourceNode(uvNode: Node): Node {
    const node = tslTexture(this.placeholder, uvNode)
    this.analysisSourceNodes.push(node)
    return node
  }

  private trackDetailSourceNode(uvNode: Node): Node {
    const node = tslTexture(this.placeholder, uvNode)
    this.detailSourceNodes.push(node)
    return node
  }

  private trackAtlasTextureNode(uvNode: Node): Node {
    const node = tslTexture(this.atlas?.texture ?? this.placeholder, uvNode)
    this.atlasTextureNodes.push(node)
    return node
  }

  private trackFeatureTextureNode(uvNode: Node): Node {
    const node = tslTexture(
      this.atlas?.featureTexture ?? this.placeholder,
      uvNode
    )
    this.featureTextureNodes.push(node)
    return node
  }

  private trackLutTextureNode(uvNode: Node): Node {
    const node = tslTexture(this.lutTexture ?? this.placeholder, uvNode)
    this.lutTextureNodes.push(node)
    return node
  }

  private trackAnalysisTextureNode(uvNode: Node): Node {
    return tslTexture(this.analysisPass.texture, uvNode)
  }

  private trackDetailTextureNode(uvNode: Node): Node {
    return tslTexture(this.detailPass.texture, uvNode)
  }

  private trackLayoutTextureNode(uvNode: Node): Node {
    return tslTexture(this.layoutPass.texture, uvNode)
  }

  private trackSelectTextureNode(uvNode: Node): Node {
    return tslTexture(this.selectPass.texture, uvNode)
  }

  private buildLuma(color: Node): Node {
    return dot(vec3(color), vec3(0.2126, 0.7152, 0.0722))
  }

  private buildAnalysisColorNode(): Node {
    this.analysisSourceNodes = []

    const gridSize = this.getGridSize()
    const cellUvSize = this.getCellUvSize()
    const gridUv = vec2(uv().x, float(1).sub(uv().y))
    const cellId = floor(gridUv.mul(gridSize))
    const cellOrigin = this.getGridOrigin().add(cellId.mul(cellUvSize))

    let accumulated = vec3(float(0), float(0), float(0))

    for (let row = 0; row < SUPERSAMPLE; row += 1) {
      for (let column = 0; column < SUPERSAMPLE; column += 1) {
        const offset = vec2(
          float((column + 0.5) / SUPERSAMPLE),
          float((row + 0.5) / SUPERSAMPLE)
        ).mul(cellUvSize)
        const sampleUv = clamp(
          cellOrigin.add(offset),
          vec2(float(0), float(0)),
          vec2(float(1), float(1))
        )
        const sampled = this.trackAnalysisSourceNode(sampleUv)
        accumulated = accumulated.add(sampled.rgb)
      }
    }

    const averaged = accumulated.div(float(SUPERSAMPLE * SUPERSAMPLE))

    return vec4(averaged, float(1))
  }

  /** Luma of every DETAIL×DETAIL sub-cell, for structure matching. */
  private buildDetailColorNode(): Node {
    this.detailSourceNodes = []

    const detailSize = this.getGridSize().mul(float(DETAIL))
    const subUvSize = this.getCellUvSize().div(float(DETAIL))
    const gridUv = vec2(uv().x, float(1).sub(uv().y))
    const subId = floor(gridUv.mul(detailSize))
    const subOrigin = this.getGridOrigin().add(subId.mul(subUvSize))

    let accumulated = float(0)

    for (const [ox, oy] of [
      [0.25, 0.25],
      [0.75, 0.25],
      [0.25, 0.75],
      [0.75, 0.75],
    ]) {
      const sampleUv = clamp(
        subOrigin.add(vec2(float(ox ?? 0), float(oy ?? 0)).mul(subUvSize)),
        vec2(float(0), float(0)),
        vec2(float(1), float(1))
      )
      accumulated = accumulated.add(
        this.buildLuma(this.trackDetailSourceNode(sampleUv).rgb)
      )
    }

    return vec4(accumulated.mul(float(0.25)), float(0), float(0), float(1))
  }

  private sampleCellColor(cellIndex: Node, gridSize: Node): Node {
    return this.trackAnalysisTextureNode(
      clamp(
        cellIndex.add(vec2(0.5, 0.5)).div(gridSize),
        vec2(float(0), float(0)),
        vec2(float(1), float(1))
      )
    )
  }

  private buildLayoutColorNode(): Node {
    const gridSize = this.getGridSize()
    const gridUv = vec2(uv().x, float(1).sub(uv().y))
    const cellId = floor(gridUv.mul(gridSize))

    let chosen = float(0)
    let found = float(0)

    for (let level = this.currentBreakLevels; level >= 1; level -= 1) {
      const span = float(2 ** level)
      const half = span.mul(float(0.5))
      const blockOrigin = floor(cellId.div(span)).mul(span)

      let lowest = float(1e9)
      let highest = float(-1e9)

      for (const [qx, qy] of [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ]) {
        const quadrant = blockOrigin.add(
          vec2(float((qx ?? 0) + 0.5), float((qy ?? 0) + 0.5)).mul(half)
        )
        const luma = this.buildLuma(
          this.sampleCellColor(quadrant, gridSize).rgb
        )
        lowest = min(lowest, luma)
        highest = max(highest, luma)
      }

      const flat = step(highest.sub(lowest), this.breakThresholdUniform)
      const take = flat.mul(float(1).sub(found))
      chosen = mix(chosen, float(level), take)
      found = max(found, flat)
    }

    return vec4(chosen, float(0), float(0), float(1))
  }

  private buildShapedSignal(color: Node): Node {
    return this.buildShapedLuma(this.buildLuma(color))
  }

  private buildShapedLuma(luma: Node): Node {
    const range = max(
      this.signalWhitePointUniform.sub(this.signalBlackPointUniform),
      float(0.001)
    )
    const shaped = clamp(
      luma.sub(this.signalBlackPointUniform).div(range),
      float(0),
      float(1)
    )

    return mix(shaped, float(1).sub(shaped), this.invertUniform)
  }

  /** Unit = the cell, or the merged block it belongs to when Break Grid is on. */
  private resolveUnit(
    cell: Node,
    gridSize: Node
  ): { origin: Node; span: Node } {
    if (this.currentBreakLevels === 0) {
      return { origin: cell, span: float(1) }
    }

    const cellTexelUv = clamp(
      cell.add(vec2(0.5, 0.5)).div(gridSize),
      vec2(float(0), float(0)),
      vec2(float(1), float(1))
    )
    const level = float(this.trackLayoutTextureNode(cellTexelUv).r)
    const span = pow(float(2), level)

    return { origin: floor(cell.div(span)).mul(span), span }
  }

  private buildHash(seed: Node): Node {
    return fract(sin(dot(seed, vec2(127.1, 311.7))).mul(float(43758.5453123)))
  }

  private buildLutIndex(signal: Node): Node {
    const texel = signal
      .mul(float((RAMP_LUT_SIZE - 1) / RAMP_LUT_SIZE))
      .add(float(0.5 / RAMP_LUT_SIZE))

    return float(this.trackLutTextureNode(vec2(texel, float(0.5))).r)
  }

  /** Per cell: tone LUT → shuffle jitter → structure match → packed glyph index. */
  private buildSelectColorNode(): Node {
    this.featureTextureNodes = []
    this.lutTextureNodes = []

    const structureEnabled = this.currentStructureEnabled

    // Fn + Loop keep the node graph small: an unrolled candidate chain makes
    // TSL's cache-key traversal exponential (66 s compiles at 12 candidates).
    const select = Fn(() => {
      const gridSize = this.getGridSize()
      const gridUv = vec2(uv().x, float(1).sub(uv().y))
      const cellId = floor(gridUv.mul(gridSize))
      const unit = this.resolveUnit(cellId, gridSize)
      const unitOrigin = unit.origin.toVar()
      const unitSpan = unit.span.toVar()
      const unitCenter = unitOrigin.add(unitSpan.mul(0.5).sub(0.5))
      const color = this.sampleCellColor(unitCenter, gridSize).rgb
      const signal = this.buildShapedSignal(color).toVar()
      const lastIndex = this.rampCountUniform.sub(float(1)).toVar()

      const base = this.buildLutIndex(signal)
      const hash = this.buildHash(
        unitOrigin.add(this.shuffleSeedUniform.mul(float(0.731)))
      )
      const jitter = hash
        .mul(float(2))
        .sub(float(1))
        .mul(this.shuffleUniform)
        .mul(float(SHUFFLE_WINDOW))
        .mul(this.rampCountUniform)
      const index = clamp(
        floor(base.add(jitter).add(float(0.5))),
        float(0),
        lastIndex
      ).toVar()

      if (structureEnabled) {
        const detailSize = gridSize.mul(float(DETAIL))
        const raw: Node[] = []
        let sum: Node = float(0)

        for (let row = 0; row < DETAIL; row += 1) {
          for (let column = 0; column < DETAIL; column += 1) {
            const subCell = unitOrigin
              .mul(float(DETAIL))
              .add(vec2(float(column + 0.5), float(row + 0.5)).mul(unitSpan))
            const subUv = clamp(
              subCell.div(detailSize),
              vec2(float(0), float(0)),
              vec2(float(1), float(1))
            )
            const luma = float(this.trackDetailTextureNode(subUv).r)
            const shaped = this.buildShapedLuma(luma).toVar()
            raw.push(shaped)
            sum = sum.add(shaped)
          }
        }

        const mean = sum.div(float(DETAIL_DIM)).toVar()
        const centered = raw.map((value) => value.sub(mean).toVar())
        let norm: Node = float(0)
        for (const value of centered) {
          norm = norm.add(value.mul(value))
        }
        const inverseNorm = float(1)
          .div(max(sqrt(norm), float(1e-4)))
          .toVar()
        const feature: Node[] = []
        for (let texel = 0; texel < ASCII_FEATURE_TEXELS; texel += 1) {
          const base4 = texel * 4
          feature.push(
            vec4(
              centered[base4] ?? float(0),
              centered[base4 + 1] ?? float(0),
              centered[base4 + 2] ?? float(0),
              centered[base4 + 3] ?? float(0)
            )
              .mul(inverseNorm)
              .toVar()
          )
        }

        const window = max(
          float(1),
          this.structureUniform
            .mul(float(STRUCTURE_WINDOW))
            .mul(this.rampCountUniform)
        ).toVar()
        const featureWidth = this.atlasColumnsUniform
          .mul(float(ASCII_FEATURE_TEXELS))
          .toVar()
        const bestScore = float(-1e9).toVar()
        const bestIndex = index.toVar()

        Loop(STRUCTURE_CANDIDATES, ({ i }: { i: Node }) => {
          const offset = float(i)
            .div(float(STRUCTURE_CANDIDATES - 1))
            .mul(float(2))
            .sub(float(1))
            .toVar()
          const candidateIndex = clamp(
            floor(index.add(offset.mul(window)).add(float(0.5))),
            float(0),
            lastIndex
          ).toVar()
          const row = floor(
            candidateIndex.div(this.atlasColumnsUniform)
          ).toVar()
          const column = candidateIndex
            .sub(row.mul(this.atlasColumnsUniform))
            .toVar()
          const rowUv = row.add(float(0.5)).div(this.atlasRowsUniform).toVar()
          const score = abs(offset).mul(float(-0.02)).toVar()

          for (let texel = 0; texel < ASCII_FEATURE_TEXELS; texel += 1) {
            const columnUv = column
              .mul(float(ASCII_FEATURE_TEXELS))
              .add(float(texel + 0.5))
              .div(featureWidth)
            const packed = this.trackFeatureTextureNode(vec2(columnUv, rowUv))
            score.addAssign(dot(vec4(packed), feature[texel] ?? vec4(0)))
          }

          const better = step(bestScore, score).toVar()
          bestIndex.assign(mix(bestIndex, candidateIndex, better))
          bestScore.assign(max(bestScore, score))
        })

        index.assign(bestIndex)
      }

      const hi = floor(index.div(float(INDEX_HI_SCALE)))
      const lo = index.sub(hi.mul(float(INDEX_HI_SCALE)))

      return vec4(hi, lo, float(0), float(1))
    })

    return select()
  }

  private buildGlyphAtlasUv(localCellUv: Node, charIndex: Node): Node {
    const column = charIndex.sub(
      floor(charIndex.div(this.atlasColumnsUniform)).mul(
        this.atlasColumnsUniform
      )
    )
    const row = floor(charIndex.div(this.atlasColumnsUniform))
    const uvInCell = vec2(
      this.atlasPadXUniform.add(localCellUv.x.mul(this.atlasInnerXUniform)),
      this.atlasPadYUniform.add(localCellUv.y.mul(this.atlasInnerYUniform))
    )

    return vec2(
      column.add(uvInCell.x).div(this.atlasColumnsUniform),
      row.add(uvInCell.y).div(this.atlasRowsUniform)
    )
  }

  private buildCharacterMask(
    localCellUv: Node,
    charIndex: Node,
    blockSpan: Node
  ): Node {
    const insideCell = step(float(0), localCellUv.x)
      .mul(step(localCellUv.x, float(1)))
      .mul(step(float(0), localCellUv.y))
      .mul(step(localCellUv.y, float(1)))
    const safeUv = clamp(
      localCellUv,
      vec2(float(0), float(0)),
      vec2(float(1), float(1))
    )
    const signedDistance = float(
      this.trackAtlasTextureNode(this.buildGlyphAtlasUv(safeUv, charIndex)).r
    )
    const atlasPixelsInside = signedDistance
      .sub(float(0.5))
      .mul(this.sdfRadiusUniform)
      .add(this.boldnessUniform.mul(float(2)))
    const atlasToDevice = this.cellPixelHeightUniform
      .mul(blockSpan)
      .mul(this.renderScaleUniform)
      .div(this.atlasInnerHeightUniform)

    return clamp(
      smoothstep(float(-0.5), float(0.5), atlasPixelsInside.mul(atlasToDevice)),
      float(0),
      float(1)
    ).mul(insideCell)
  }

  private buildCellSample(
    candidateCell: Node,
    gridUv: Node
  ): {
    color: Node
    mask: Node
  } {
    const gridSize = this.getGridSize()
    const cellUvSize = this.getCellUvSize()
    const gridOrigin = this.getGridOrigin()
    const unit = this.resolveUnit(candidateCell, gridSize)
    const unitOriginCell = unit.origin
    const unitSpan = unit.span

    const unitCenterCell = unitOriginCell.add(unitSpan.mul(0.5).sub(0.5))
    const color = this.sampleCellColor(unitCenterCell, gridSize).rgb
    const signal = this.buildShapedSignal(color)

    const selectUv = clamp(
      unitOriginCell.add(vec2(0.5, 0.5)).div(gridSize),
      vec2(float(0), float(0)),
      vec2(float(1), float(1))
    )
    const packed = this.trackSelectTextureNode(selectUv)
    const charIndex = floor(
      float(packed.r)
        .mul(float(INDEX_HI_SCALE))
        .add(float(packed.g))
        .add(float(0.5))
    )

    let unitOriginUv = gridOrigin.add(unitOriginCell.mul(cellUvSize))

    if (this.currentRowWarpEnabled) {
      const rowOffset = signal
        .sub(float(0.5))
        .mul(this.rowWarpUniform)
        .mul(cellUvSize.x)
        .mul(unitSpan)
      unitOriginUv = vec2(unitOriginUv.x.add(rowOffset), unitOriginUv.y)
    }

    const localCellUv = gridUv.sub(unitOriginUv).div(cellUvSize.mul(unitSpan))
    const mask = this.buildCharacterMask(localCellUv, charIndex, unitSpan)

    return { color, mask }
  }

  protected override buildEffectNode(): Node {
    if (!this.rampCountUniform) {
      return this.inputNode
    }

    this.atlasTextureNodes = []

    const gridUv = vec2(uv().x, float(1).sub(uv().y))
    const cellUvSize = this.getCellUvSize()
    const gridOrigin = this.getGridOrigin()
    const baseCell = floor(gridUv.sub(gridOrigin).div(cellUvSize))

    const candidateSteps = this.currentRowWarpEnabled ? [-1, 0, 1] : [0]

    let bestMask: Node = float(0)
    let bestColor: Node = vec3(float(0), float(0), float(0))

    for (const stepX of candidateSteps) {
      const candidate =
        stepX === 0 ? baseCell : baseCell.add(vec2(float(stepX), float(0)))
      const sample = this.buildCellSample(candidate, gridUv)
      const stronger = step(bestMask, sample.mask)
      bestColor = mix(bestColor, sample.color, stronger)
      bestMask = max(bestMask, sample.mask)
    }

    const monoTint = vec3(
      this.monoRedUniform,
      this.monoGreenUniform,
      this.monoBlueUniform
    )
    const monoSignal = this.buildShapedSignal(bestColor)
    const monoColor = monoTint.mul(monoSignal)
    const glyphColor = mix(monoColor, bestColor, this.sourceMixUniform)
    const backgroundColor = bestColor
      .mul(this.bgOpacityUniform)
      .mul(this.sourceMixUniform)

    return vec4(mix(backgroundColor, glyphColor, bestMask), float(1))
  }

  private scheduleStructuralRebuild(): void {
    const renderer = this.lastRenderer

    if (!renderer) {
      this.analysisPass.setColorNode(this.buildAnalysisColorNode())
      this.detailPass.setColorNode(this.buildDetailColorNode())
      this.layoutPass.setColorNode(this.buildLayoutColorNode())
      this.selectPass.setColorNode(this.buildSelectColorNode())
      this.rebuildEffectNode()
      return
    }

    const generation = ++this.rebuildGeneration
    const previousAnalysis = this.analysisSourceNodes
    const previousDetail = this.detailSourceNodes
    const previousAtlas = this.atlasTextureNodes
    const previousFeatures = this.featureTextureNodes
    const previousLut = this.lutTextureNodes

    const jobs = [
      this.analysisPass.setColorNodeAsync(
        this.buildAnalysisColorNode(),
        renderer
      ),
      this.detailPass.setColorNodeAsync(this.buildDetailColorNode(), renderer),
      this.layoutPass.setColorNodeAsync(this.buildLayoutColorNode(), renderer),
      this.selectPass.setColorNodeAsync(this.buildSelectColorNode(), renderer),
      this.swapEffectNodeAsync(this.buildEffectNode()),
    ]

    const nextAnalysis = this.analysisSourceNodes
    const nextDetail = this.detailSourceNodes
    const nextAtlas = this.atlasTextureNodes
    const nextFeatures = this.featureTextureNodes
    const nextLut = this.lutTextureNodes

    this.analysisSourceNodes = [...previousAnalysis, ...nextAnalysis]
    this.detailSourceNodes = [...previousDetail, ...nextDetail]
    this.atlasTextureNodes = [...previousAtlas, ...nextAtlas]
    this.featureTextureNodes = [...previousFeatures, ...nextFeatures]
    this.lutTextureNodes = [...previousLut, ...nextLut]

    void Promise.all(jobs)
      .then(() => {
        if (generation !== this.rebuildGeneration) {
          return
        }

        this.analysisSourceNodes = nextAnalysis
        this.detailSourceNodes = nextDetail
        this.atlasTextureNodes = nextAtlas
        this.featureTextureNodes = nextFeatures
        this.lutTextureNodes = nextLut
      })
      .catch(() => {
        if (generation === this.rebuildGeneration) {
          this.analysisPass.setColorNode(this.buildAnalysisColorNode())
          this.detailPass.setColorNode(this.buildDetailColorNode())
          this.layoutPass.setColorNode(this.buildLayoutColorNode())
          this.selectPass.setColorNode(this.buildSelectColorNode())
          this.rebuildEffectNode()
        }
      })
  }

  private resolveCharset(value: unknown): AsciiCharset {
    return typeof value === "string" &&
      (value === "custom" || value in ASCII_CHARSETS)
      ? (value as AsciiCharset)
      : "light"
  }

  private getActiveChars(): string {
    return this.currentCharset === "custom"
      ? this.currentCustomChars || " "
      : (ASCII_CHARSETS[this.currentCharset] ?? DEFAULT_ASCII_CHARS)
  }

  private getAtlasOptions(): AsciiAtlasOptions {
    return {
      autoSort: true,
      cellAspect: 0,
      chars: this.getActiveChars(),
      edgeChars: DEFAULT_EDGE_CHARS,
      fontFamily: this.currentFontFamily,
      fontWeight: this.currentFontWeight,
    }
  }

  private rebuildLut(): void {
    if (!this.atlas) {
      return
    }

    const previous = this.lutTexture
    this.lutTexture = buildRampLutTexture(this.atlas, this.currentTonemap)

    for (const node of this.lutTextureNodes) {
      node.value = this.lutTexture
    }

    if (previous) {
      this.retiredAtlasTextures.push(previous)
      this.framesSinceAtlasSwap = 0
    }
  }

  private rebuildAtlas(): void {
    if (typeof document === "undefined") {
      return
    }

    const options = this.getAtlasOptions()
    const previousTexture = this.atlas?.texture
    const previousFeatures = this.atlas?.featureTexture
    const atlas = buildAsciiAtlas(options)
    this.atlas = atlas

    for (const node of this.atlasTextureNodes) {
      node.value = atlas.texture
    }
    for (const node of this.featureTextureNodes) {
      node.value = atlas.featureTexture
    }

    if (previousTexture) {
      this.retiredAtlasTextures.push(previousTexture)
    }
    if (previousFeatures) {
      this.retiredAtlasTextures.push(previousFeatures)
    }
    this.framesSinceAtlasSwap = 0

    this.atlasColumnsUniform.value = atlas.columns
    this.atlasInnerXUniform.value = atlas.innerFractionX
    this.atlasInnerYUniform.value = atlas.innerFractionY
    this.atlasPadXUniform.value = atlas.padFractionX
    this.atlasPadYUniform.value = atlas.padFractionY
    this.atlasRowsUniform.value = atlas.rows
    this.cellAspectUniform.value = atlas.cellAspect
    this.rampCountUniform.value = atlas.rampCount
    this.sdfRadiusUniform.value = atlas.sdfRadius

    this.rebuildLut()
    this.ensureFontLoaded(options)
  }

  private ensureFontLoaded(options: AsciiAtlasOptions): void {
    if (isAsciiFontReady(options)) {
      return
    }

    this.fontLoadToken += 1
    const token = this.fontLoadToken

    void loadAsciiFont(options).then(() => {
      if (token !== this.fontLoadToken) {
        return
      }

      this.rebuildAtlas()
    })
  }
}
