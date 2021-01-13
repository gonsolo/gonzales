final class NamedTexture<T: Initializable>: Texture<T> {

        init(name: String) {
                self.name = name
        }

        override func evaluate(at: Interaction) -> T {
                return T.init()
        }

        let name: String
}

