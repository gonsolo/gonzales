struct SurfaceInteraction: Interaction, Sendable {

        func evaluateDistributionFunction(wi: Vector) -> RgbSpectrum {
                let reflected = bsdf.evaluateWorld(wo: wo, wi: wi)
                let dot = absDot(wi, Vector(normal: shadingNormal))
                let scatter = reflected * dot
                return scatter
        }

        func sampleDistributionFunction(sampler: any Sampler) -> BsdfSample {
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
        //var material: MaterialIndex = -1

        var material: Material = Material.diffuse(
                Diffuse(
                        reflectance: Texture.rgbSpectrumTexture(
                                RgbSpectrumTexture.constantTexture(ConstantTexture(value: white)))))
        var mediumInterface: MediumInterface? = nil
        var bsdf: any GlobalBsdf = DummyBsdf()
}

extension SurfaceInteraction: CustomStringConvertible {
        var description: String {
                return "[pos: \(position) n: \(normal) wo: \(wo) ]"
        }
}
