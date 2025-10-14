protocol DistributionModel {
        func evaluateDistributionFunction(wo: Vector, wi: Vector, normal: Normal) -> RgbSpectrum
        func sampleDistributionFunction(wo: Vector, normal: Normal, sampler: RandomSampler) -> BsdfSample
        func evaluateProbabilityDensity(wo: Vector, wi: Vector) -> FloatX
}

struct SurfaceInteraction: Interaction, Sendable {

        func getBsdf() -> GlobalBsdfType {
                return material.getBsdf(interaction: self)
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
        //var material: MaterialIndex = -1

        var material: Material = Material.diffuse(
                Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white)))))
        var mediumInterface: MediumInterface? = nil
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return
                        "[" + "valid: \(valid) " + "pos: \(position) " + "n: \(normal) "
                        + "shadingNormal: \(shadingNormal) " + "wo: \(wo) " + "dpdu: \(dpdu) " + "uv: \(uv) "
                        + "faceIndex: \(faceIndex) " + "areaLight: \(areaLight as Optional) "
                        + "material: \(material) " + "mediumInterface: \(mediumInterface as Optional) "
                        + "]"
        }
}

extension SurfaceInteraction: Equatable {
        static func == (lhs: SurfaceInteraction, rhs: SurfaceInteraction) -> Bool {
                return
                        lhs.valid == rhs.valid && lhs.position == rhs.position && lhs.normal == rhs.normal
                        && lhs.shadingNormal == rhs.shadingNormal && lhs.wo == rhs.wo && lhs.dpdu == rhs.dpdu
                        && lhs.uv == rhs.uv && lhs.faceIndex == rhs.faceIndex
                        && lhs.areaLight == rhs.areaLight  //&&
                //lhs.material == rhs.material &&
                //lhs.mediumInterface == rhs.mediumInterface &&
                //lhs.bsdf == rhs.bsdf
        }
}
