final class Mirror: Material {

        init(kr: Texture<Spectrum>) { self.kr = kr }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                let reflectance = kr.evaluate(at: interaction)
                bsdf.add(bxdf: SpecularReflection(reflectance: reflectance, fresnel: FresnelNop()))
		return (bsdf, nil)
        }

        var kr: Texture<Spectrum>
}

func createMirror(parameters: ParameterDictionary) throws -> Mirror {
        let kr = try parameters.findSpectrumTexture(name: "Kr")
        return Mirror(kr: kr)
}

