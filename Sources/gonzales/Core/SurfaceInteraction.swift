struct SurfaceInteraction: Interaction, Sendable {

        init(   valid: Bool = false,
                position: Point = Point(),
                normal: Normal = Normal(),
                shadingNormal: Normal = Normal(),
                wo: Vector = Vector(),
                dpdu: Vector = Vector(),
                uv: Point2f = Point2f(),
                faceIndex: Int = 0, 
                material: Material = Material.diffuse(
                Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white)))))
             ) {
                old = InnerSurfaceInteraction(
                        valid: valid,
                        position: position,
                        normal: normal,
                        shadingNormal: shadingNormal,
                        wo: wo,
                        dpdu: dpdu,
                        uv: uv,
                        faceIndex: faceIndex,
                        material: material)
                new =  .surface(old)

                assert(old.position == new.position)
        }

        func spawnRay(to: Point) -> (ray: Ray, tHit: FloatX) {
                let o = old.spawnRay(to: to)
                let n = new.spawnRay(to: to)
                assert(o.ray == n.ray && o.tHit == n.tHit)
                return o
        }

        func spawnRay(inDirection direction: Vector) -> Ray {
                return old.spawnRay(inDirection: direction)
        }

        func evaluateDistributionFunction(wi: Vector) -> RgbSpectrum {
                return old.evaluateDistributionFunction(wi: wi)
        }

        func sampleDistributionFunction(sampler: RandomSampler) -> BsdfSample {
                return old.sampleDistributionFunction(sampler: sampler)
        }

        func evaluateProbabilityDensity(wi: Vector) -> FloatX {
                return old.evaluateProbabilityDensity(wi: wi)
        }

        var dpdu: Vector {
                get {
                        assert(old.dpdu == new.dpdu)
                        return old.dpdu
                }
                set(newDpdu) {
                        old.dpdu = newDpdu
                        new.dpdu = newDpdu
                }
        }
        var faceIndex: Int { 
                get {
                        return old.faceIndex
                }
                set(newFaceIndex) {
                        old.faceIndex = newFaceIndex
                }
        }
        var normal: Normal { 
                get {
                        return old.normal
                }
                set(newNormal) {
                        old.normal = newNormal
                }
        }

        var position: Point { 
                get {
                        assert(old.position == new.position)
                        return old.position
                }
                set(newPosition) {
                        old.position = newPosition
                        new.position = newPosition
                        assert(old.position == new.position)
                }
        }
        var shadingNormal: Normal {
                get {
                        return old.shadingNormal
                }
                set(newShadingNormal) {
                        old.shadingNormal = newShadingNormal
                }
        }
        var uv: Point2f { 
                get {
                        return old.uv
                }
                set(newUv) {
                        old.uv = newUv
                }
        }
        var wo: Vector {
                get {
                        return old.wo
                }
                set(newWo) {
                        old.wo = newWo
                }
        }

        var valid: Bool { 
                get {
                        return old.valid
                }
                set(newValid) {
                        old.valid = newValid
                }
        }
        var material: Material {
                get {
                        return old.material
                }
                set(newMaterial) {
                        old.material = newMaterial
                }
        }
        var mediumInterface: MediumInterface? {
                get {
                        return old.mediumInterface
                }
                set(newMediumInterface) {
                        old.mediumInterface = newMediumInterface
                }
        }
        var areaLight: AreaLight? {
                get {
                        return old.areaLight
                }
                set(newAreaLight) {
                        old.areaLight = newAreaLight
                }
        }
        var bsdf: GlobalBsdfType  {
                get {
                        return old.bsdf
                }
        }

        mutating func setBsdf() {
                old.setBsdf()
        }

        var old: InnerSurfaceInteraction
        var new: InteractionType
}

struct InnerSurfaceInteraction: Interaction, Sendable {

        init(   valid: Bool = false,
                position: Point = Point(),
                normal: Normal = Normal(),
                shadingNormal: Normal = Normal(),
                wo: Vector = Vector(),
                dpdu: Vector = Vector(),
                uv: Point2f = Point2f(),
                faceIndex: Int = 0, 
                material: Material = Material.diffuse(
                Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white)))))
             ) {
                self.valid = valid
                self.position = position
                self.normal = normal
                self.shadingNormal = shadingNormal
                self.wo = wo
                self.dpdu = dpdu
                self.uv = uv
                self.faceIndex = faceIndex
                self.material = material
        }

        func evaluateDistributionFunction(wi: Vector) -> RgbSpectrum {
                let reflected = bsdf.evaluateWorld(wo: wo, wi: wi)
                let dot = absDot(wi, Vector(normal: shadingNormal))
                let scatter = reflected * dot
                return scatter
        }

        func sampleDistributionFunction(sampler: RandomSampler) -> BsdfSample {
                var (bsdfSample, _) = bsdf.sampleWorld(wo: wo, u: sampler.get3D())
                bsdfSample.estimate *= absDot(bsdfSample.incoming, shadingNormal)
                return bsdfSample
        }

        func evaluateProbabilityDensity(wi: Vector) -> FloatX {
                return bsdf.probabilityDensityWorld(wo: wo, wi: wi)
        }

        mutating func setBsdf() {
                bsdf = material.getBsdf(interaction: self)
        }

        var valid = false
        var position = Point()
        var normal = Normal()
        var shadingNormal = Normal()
        var wo = Vector()
        var dpdu = Vector()
        var uv = Point2f()
        var faceIndex = 0
        var areaLight: AreaLight? = nil

        var material: Material = Material.diffuse(
                Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white)))))
        var mediumInterface: MediumInterface? = nil
        var bsdf: GlobalBsdfType = .dummyBsdf(DummyBsdf())
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return "[pos: \(position) n: \(normal) wo: \(wo) ]"
        }
}
