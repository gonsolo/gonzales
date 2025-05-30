struct FloatMixTexture {

        func evaluateFloat(at interaction: any Interaction) -> Float {
                let value0 = textures.0.evaluateFloat(at: interaction)
                let value1 = textures.1.evaluateFloat(at: interaction)
                return lerp(with: amount, between: value0, and: value1)
        }

        let textures: (FloatTexture, FloatTexture)
        let amount: FloatX
}
