public typealias Matrix = FloatMatrix<MatrixBacking<FloatX>>
/*
public typealias AffineMatrix = FloatMatrix<AffineMatrixBacking<FloatX>>
typealias SmallMatrix = FloatMatrix<MatrixBacking<Float32>>
typealias SmallAffineMatrix = FloatMatrix<AffineMatrixBacking<Float32>>
extension Matrix {
        init(_ m: SmallMatrix) {
                self.init()
                for x in 0..<4 {
                        for y in 0..<4 {
                                //self.m[x][y] = FloatX(m.m[x][y])
                                self.backing[x, y] = FloatX(backing[x, y])
                        }
                }
        }
}

extension SmallMatrix {
        init(_ m: Matrix) {
                self.init()
                for x in 0..<4 {
                        for y in 0..<4 {
                                //self.m[x][y] = Float16(m.m[x][y])
                                self.backing[x, y] = Float16(m.backing[x, y])
                        }
                }
        }
}

*/

