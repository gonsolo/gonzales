final class SpecularTransmission: BxDF {

        init(t: Spectrum, etaA: FloatX, etaB: FloatX) {
                self.t = t
                self.etaA = etaA
                self.etaB = etaB
                self.fresnel = FresnelDielectric(etaI: etaA, etaT: etaB)
        }

        func evaluate(wo: Vector, wi: Vector) -> Spectrum { return black }

        func sample(wo: Vector, u: Point2F) -> (Spectrum, Vector, Float) {
                // TODO: eta
                //let normal = faceforward(normal: Normal(x: 0, y: 0, z: 1), comparedTo: wo)
                //guard let wi = refract(wi: wo, normal: normal, eta: 1) else {
                //        return (black, nullVector, 0)
                //}
                unimplemented() // spectrum, pdf
                //return (black, wi, 0)
        }

        func probabilityDensity(wo: Vector, wi: Vector) -> FloatX { return 0 }

        func albedo() -> Spectrum { return white }

        let t: Spectrum
        let etaA: FloatX
        let etaB: FloatX
        let fresnel: FresnelDielectric
}
