struct CameraSample {

        init(film: TwoRandomVariables = (0, 0), lens: TwoRandomVariables = (0, 0)) {
                self.film = film
                self.lens = lens
        }

        let film: TwoRandomVariables
        let lens: TwoRandomVariables
}

extension CameraSample: Equatable {
        static func == (lhs: CameraSample, rhs: CameraSample) -> Bool {
                return
                        lhs.film == rhs.film &&
                        lhs.lens == rhs.lens
        }
}
