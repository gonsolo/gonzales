final class Glass: Material {

        init(reflectance: Texture<Spectrum>, transmittance: Texture<Spectrum>, eta: Texture<FloatX>) {
                self.reflectance = reflectance
                self.transmittance = transmittance
                self.eta = eta 
        }

        func computeScatteringFunctions(interaction: Interaction) -> (BSDF, BSSRDF?) {
                var bsdf = BSDF(interaction: interaction)
                let eta = self.eta.evaluate(at: interaction)
                bsdf.eta = eta
                let reflectance = self.reflectance.evaluate(at: interaction)
                let transmittance = self.transmittance.evaluate(at: interaction)
                if reflectance.isBlack && transmittance.isBlack { return (bsdf, nil) }
                let specular = FresnelSpecular(reflectance: reflectance, transmittance: transmittance, etaA: 1, etaB: eta)
                bsdf.add(bxdf: specular)
                return (bsdf, nil)
        }

        var reflectance: Texture<Spectrum>
        var transmittance: Texture<Spectrum>
        var eta: Texture<FloatX>
}

func createGlass(parameters: ParameterDictionary) throws -> Glass {
        let kr = try parameters.findSpectrumTexture(name: "Kr", else: Spectrum(intensity: 1))
        let kt = try parameters.findSpectrumTexture(name: "Kt", else: Spectrum(intensity: 1))
        let eta = try parameters.findFloatXTexture(name: "eta", else: 1.5)
        return Glass(reflectance: kr, transmittance: kt, eta: eta)
}

