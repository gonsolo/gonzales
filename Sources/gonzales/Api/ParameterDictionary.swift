/// A collection of parameters.
///
/// Every parameter has a type an a value (or a collection of values).

typealias ParameterDictionary = [String: any Parameter]

extension ParameterDictionary {

        func findSpectrum(name: String, else spectrum: (any Spectrum)? = nil)
                throws -> (any Spectrum)?
        {
                let spectra = try findSpectra(name: name)
                if spectra.isEmpty {
                        return spectrum
                } else {
                        return spectra[0]
                }
        }

        func findOnePoint(name: String, else preset: Point) throws -> Point {
                let points = try findPoints(name: name)
                if points.isEmpty {
                        return preset
                } else {
                        return points[0]
                }
        }

        func findString(called name: String) throws -> String? {
                let strings = try findStrings(name: name)
                if strings.isEmpty {
                        return nil
                } else {
                        return strings[0]
                }
        }

        func findTexture(name: String) throws -> String {
                return try findString(called: name) ?? ""
        }

        @MainActor
        func findRgbSpectrumTexture(
                name: String,
                else spectrum: RgbSpectrum = RgbSpectrum(intensity: 1)
        )
                throws -> RgbSpectrumTexture
        {
                let textureName = try findTexture(name: name)
                if textureName != "" {
                        guard let texture = state.textures[textureName] else {
                                fatalError("Could not find texture!")
                        }
                        switch texture {
                        case .floatTexture:
                                warning("Could not find texture \(textureName)")
                                let constantTexture = ConstantTexture<RgbSpectrum>(value: red)
                                let rgbSpectrumTexture = RgbSpectrumTexture.constantTexture(constantTexture)
                                return rgbSpectrumTexture
                        case .rgbSpectrumTexture(let rgbSpectrumTexture):
                                return rgbSpectrumTexture
                        }
                } else {
                        guard let spectrum = try findSpectrum(name: name, else: spectrum) else {
                                fatalError()
                        }
                        guard let rgbSpectrum = spectrum as? RgbSpectrum else {
                                fatalError()
                        }
                        let constantTexture = ConstantTexture(value: rgbSpectrum)
                        return RgbSpectrumTexture.constantTexture(constantTexture)
                }
        }

        @MainActor
        func findFloatXTexture(name: String, else value: FloatX = 1.0) throws -> FloatTexture {
                let textureName = try findTexture(name: name)
                if textureName != "" {
                        guard let texture = state.textures[textureName] else {
                                print("No named texture \"\(textureName)\"")
                                fatalError()
                        }
                        switch texture {
                        case .floatTexture(let floatTexture):
                                return floatTexture
                        case .rgbSpectrumTexture:
                                print("No named float texture \"\(textureName)\"")
                                fatalError()
                        }
                } else {
                        let value = try findOneFloatX(called: name, else: value)
                        return FloatTexture.constantTexture(ConstantTexture<FloatX>(value: value))
                }
        }

        func findOneBool(called name: String, else preset: Bool) throws -> Bool {
                let bools = try findBools(name: name)
                if bools.isEmpty {
                        return preset
                } else {
                        return bools[0]
                }
        }

        func findOneFloatXOptional(called name: String) throws -> FloatX? {
                let floats = try findFloatXs(called: name)
                if floats.isEmpty {
                        return nil
                } else {
                        return floats[0]
                }
        }

        func findOneFloatX(called name: String, else preset: FloatX) throws -> FloatX {
                let floats = try findFloatXs(called: name)
                if floats.isEmpty {
                        return preset
                } else {
                        return floats[0]
                }
        }

        func findOneInt(called name: String, else preset: Int) throws -> Int {
                let ints = try findInts(name: name)
                if ints.isEmpty {
                        return preset
                } else {
                        return ints[0]
                }
        }

        func findFloatXs(called name: String) throws -> [FloatX] {
                guard let floats = self[name] as? [FloatX] else {
                        return []
                }
                return floats
        }

        func findPoints(name: String) throws -> [Point] {
                guard let points = self[name] as? [Point] else {
                        return []
                }
                return points
        }

        func findInts(name: String) throws -> [Int] {
                guard let ints = self[name] as? [Int] else {
                        return []
                }
                return ints
        }

        func findNormals(name: String) throws -> [Normal] {
                guard let normals = self[name] as? [Normal] else {
                        return []
                }
                return normals
        }

        func findSpectra(name: String) throws -> [any Spectrum] {
                guard let spectra = self[name] as? [any Spectrum] else {
                        return []
                }
                return spectra
        }

        func findStrings(name: String) throws -> [String] {
                guard let strings = self[name] as? [String] else {
                        return []
                }
                return strings
        }

        func findBools(name: String) throws -> [Bool] {
                guard let bools = self[name] as? [Bool] else {
                        return []
                }
                return bools
        }
}
