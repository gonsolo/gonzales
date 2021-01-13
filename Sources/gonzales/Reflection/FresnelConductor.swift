struct FresnelConductor: Fresnel {

        init(etaI: Spectrum, etaT: Spectrum, k: Spectrum) {
                self.etaI = etaI
                self.etaT = etaT
                self.k = k
        }

        // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations
        func frConductor(cosTheta: FloatX, etaI: Spectrum, etaT: Spectrum, k: Spectrum) -> Spectrum {
                let cosThetaClamped = clamp(value: cosTheta, low: -1, high: 1)
                let eta = etaT / etaI
                let etak = k / etaI
                let cosTheta2 = cosThetaClamped * cosThetaClamped
                let sinThetaI2 = 1 - cosTheta2
                let eta2 = eta * eta
                let etak2 = etak * etak
                let t0 = eta2 - etak2 - sinThetaI2
                let a2plusb2 = (t0 * t0 + 4 * eta2 * etak2).squareRoot()
                let t1 = a2plusb2 + cosTheta2
                let a = (0.5 * (a2plusb2 + t0)).squareRoot()
                let t2 = 2 * cosThetaClamped * a
                let Rs = (t1 - t2) / (t1 + t2)
                let t3 = cosTheta2 * a2plusb2 + sinThetaI2 * sinThetaI2
                let t4 = t2 * sinThetaI2
                let Rp = Rs * (t3 - t4) / (t3 + t4)
                return 0.5 * (Rp + Rs)
        }

        func evaluate(cosTheta: FloatX) -> Spectrum {
                return frConductor(cosTheta: abs(cosTheta), etaI: etaI, etaT: etaT, k: k)
        }

        let etaI: Spectrum
        let etaT: Spectrum
        let k: Spectrum
}

