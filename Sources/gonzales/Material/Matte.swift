final class Matte : Material {

        init(kd: Texture<Spectrum>) { self.kd = kd }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                let kde = kd.evaluate(at: interaction)
                bsdf.add(bxdf: LambertianReflection(reflectance: kde))
		return (bsdf, nil)
        }

        var kd: Texture<Spectrum>
}

func createMatte(parameters: ParameterDictionary) throws -> Matte {
        let kd: Texture<Spectrum> = try parameters.findSpectrumTexture(name: "Kd", else: gray)
        return Matte(kd: kd)
}

