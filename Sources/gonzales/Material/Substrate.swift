final class Substrate: Material {

        init(kd: Texture<Spectrum>, ks: Texture<Spectrum>, roughness: (FloatX, FloatX)) {
                self.kd = kd
                self.ks = ks
                self.roughness = roughness
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                let kd = self.kd.evaluate(at: interaction)
                let ks = self.ks.evaluate(at: interaction)
                let alpha = TrowbridgeReitzDistribution.getAlpha(from: roughness)
                let distribution = TrowbridgeReitzDistribution(alpha: alpha)
                let fresnelBlend = FresnelBlend(diffuseReflection: kd,
                                                glossyReflection: ks,
                                                distribution: distribution)
                bsdf.add(bxdf: fresnelBlend)
                return (bsdf, nil)
        }

        var kd: Texture<Spectrum>
        var ks: Texture<Spectrum>
        var roughness: (FloatX, FloatX)
        let remapRoughness = true
}

func createSubstrate(parameters: ParameterDictionary) throws -> Substrate {
        let kd: Texture<Spectrum> = try parameters.findSpectrumTexture(name: "Kd", else: gray)
        let ks: Texture<Spectrum> = try parameters.findSpectrumTexture(name: "Ks", else: gray)
        let uroughness = try parameters.findOneFloatX(called: "uroughness", else: 0.1)
        let vroughness = try parameters.findOneFloatX(called: "vroughness", else: 0.1)
        let roughness = (uroughness, vroughness)
        return Substrate(kd: kd, ks: ks, roughness: roughness)
}

