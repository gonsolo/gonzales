final class ConstantTexture<T>: Texture<T> {

        init(value: T) {
                self.value = value
        }

        override func evaluate(at: Interaction) -> T {
                return value
        }

        var value: T
}

extension ConstantTexture: CustomStringConvertible {

        public var description: String {
                return "[ \(value) ]"
        }
}
