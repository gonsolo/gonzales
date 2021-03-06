/**
        A collection of parameters.
        Every parameter has a type an a value (or a collection of
        values).
*/
typealias ParameterDictionary = Dictionary<String, Parameter>

extension ParameterDictionary {

        func findSpectrum(name: String, else spectrum: Spectrum? = nil) throws -> Spectrum? {
                let spectra = try findSpectra(name: name)
                if spectra.isEmpty { return spectrum }
                else { return spectra[0] }
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

        func findSpectrumTexture(name: String, else spectrum: Spectrum = Spectrum(intensity: 1)) throws -> Texture<Spectrum> {
                let textureName = try findTexture(name: name)
                if textureName != "" {
                        guard let texture = state.spectrumTextures[textureName] else {
                                warning("Coud not find texture \(textureName)")
                                return ConstantTexture<Spectrum>(value: red)
                        }
                        return texture
                } else {
                        guard let spectrum = try findSpectrum(name: name, else: spectrum) else {
                                fatalError()
                        }
                        return ConstantTexture<Spectrum>(value: spectrum)
                }
        }

        func findFloatXTexture(name: String, else value: FloatX = 1.0) throws -> Texture<FloatX> {
                let textureName = try findTexture(name: name)
                if textureName != "" {
                        guard let texture = state.floatTextures[textureName] else {
                                print("No texture")
                                fatalError()
                        }
                        return texture
                } else {
                        let value = try findOneFloatX(called: name, else: value)
                        return ConstantTexture<FloatX>(value: value)
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
/*
        // This could be used instead of findInts etc...
        func find(name: String) throws -> Parameter {
                guard let values = self[name] else {
                        return []
                }
                return values

        }
*/
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

        func findSpectra(name: String) throws -> [Spectrum] {
                guard let spectra = self[name] as? [Spectrum] else {
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

