import Foundation

actor AsciiWriter {

        func write(fileName: String, crop: Bounds2i, image: Image) async throws {

                let resolution = crop.pMax - crop.pMin
                var text = String(resolution.x) + " " + String(resolution.y) + "\n"

                func write(pixel: Pixel, at index: Int) {
                        text.append(String(pixel.light.r))
                        text.append(" ")
                        text.append(String(pixel.light.g))
                        text.append(" ")
                        text.append(String(pixel.light.b))
                        text.append(" ")
                        text.append("1.0")
                        text.append(" ")
                }

                for y in crop.pMin.y..<crop.pMax.y {
                        for x in crop.pMin.x..<crop.pMax.x {
                                let location = Point2i(x: x, y: y)
                                let pixel = await image.getPixel(atLocation: location)
                                let px = x - crop.pMin.x
                                let py = y - crop.pMin.y
                                let index = py * resolution.x * 4 + px * 4
                                write(pixel: pixel, at: index)
                        }
                }

                let url = URL(fileURLWithPath: "/home/gonsolo/" + fileName)
                print(url)
                do {
                        try text.write(to: url, atomically: true, encoding: String.Encoding.utf8)
                } catch {
                        print("Error writing file")
                }
        }
}
