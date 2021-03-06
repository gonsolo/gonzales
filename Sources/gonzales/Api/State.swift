struct UninstancedMaterial {
        let type: String
        let parameters: ParameterDictionary
}

struct State {

        init() {
                namedMaterials = [String: UninstancedMaterial]() 
                currentNamedMaterial = "None"
                namedMaterials["None"] =  UninstancedMaterial(type: "matte", parameters: ParameterDictionary())
                spectrumTextures = [String: Texture<Spectrum>]()
                floatTextures = [String: Texture<FloatX>]()
                ptexCache = PtexCache()
        }

        func createMaterial(parameters: ParameterDictionary) throws -> Material {
                var material: UninstancedMaterial!
                if currentMaterial != nil {
                        material = currentMaterial!
                } else {
                        assert(currentNamedMaterial != "")
		                    guard let named = namedMaterials[currentNamedMaterial] else {
			                          warning("The material \(currentNamedMaterial) was not defined!")
			                          return Matte(kd: ConstantTexture(value: gray))
		                    }
                        material = named
                }
                var merged = parameters
                merged.merge(material.parameters) {(current, _) in current}
                return try api.makeMaterial(name: material.type, parameters: merged)
        }

        var namedMaterials: [String: UninstancedMaterial]
        var currentNamedMaterial: String
        var currentMaterial: UninstancedMaterial?
        var areaLight = ""
        var areaLightParameters = ParameterDictionary()
        var objectName: String? = nil
        var spectrumTextures: [String: Texture<Spectrum>]
        var floatTextures: [String: Texture<FloatX>]
        let ptexCache: PtexCache
}

