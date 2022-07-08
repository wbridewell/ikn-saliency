; ikn-saliency Project by the U.S. Naval Research Laboratory.
;
; As a work of the United States Government, this source code is in the public
;  domain within the United States and is not licensed or under copyright and was
; produced at the U.S. Additionally, we waive copyright and related rights in the
;  work worldwide through the CC0 1.0 Universal public domain dedication. You can
; not use this file except in compliance with the License. A copy of the license is distributed with this project in the LICENSE.txt file.
;
;  Unless required by applicable law or agreed to in writing, software distributed
;  under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
; CONDITIONS OF ANY KIND, either express or implied. See the License for the
; specific language governing permissions and limitations under the License.

(ns ikn-saliency.utils
  ^{:doc "Various utility functions to support the saliency code."}
  (:import
    java.awt.image.BufferedImage
    [org.opencv.core Core Size Mat Point CvType Scalar]
    org.opencv.imgproc.Imgproc
    org.opencv.imgcodecs.Imgcodecs))

(defn apply-to-values
  "Apply function f to the value in map m using the additional arguments."
  [m f & args]
  (into {} (for [[k v] m] [k (apply f v args)])))

(defn mat-to-bufferedimage
  "Convert an OpenCV matrix to a Java BufferedImage."
  [matrix-image]
  (let [w (.width matrix-image)
        h (.height matrix-image)
        eight-bit-matrix (Mat.)
        data (byte-array (* w h (.channels matrix-image)))
        bi (BufferedImage. w h (if (= (.channels matrix-image) 1)
                                 BufferedImage/TYPE_BYTE_GRAY
                                 BufferedImage/TYPE_3BYTE_BGR))]

    (if (= (.channels matrix-image) 1)
      (.convertTo matrix-image eight-bit-matrix CvType/CV_8UC1)
      (.convertTo matrix-image eight-bit-matrix CvType/CV_8UC3))

    (when (= (.channels matrix-image) 3)
      (Imgproc/cvtColor eight-bit-matrix eight-bit-matrix Imgproc/COLOR_BGR2RGB))

    (.get eight-bit-matrix 0 0 data)
    (.setDataElements (.getRaster bi) 0 0 w h data)
    bi))

(defn display-image
  "Display a BufferedImage in a frame with the given width and height (default: 600x800)."
  ([img]
   (display-image img 600 800))
  ([img w h]
   (javax.swing.SwingUtilities/invokeAndWait
    (fn []
      (let [frame (javax.swing.JFrame.)
            lblimage (javax.swing.JLabel. (javax.swing.ImageIcon. img))]
        (.add (.getContentPane frame) lblimage java.awt.BorderLayout/CENTER)
        (.setDefaultCloseOperation frame javax.swing.WindowConstants/DISPOSE_ON_CLOSE)
        (.setSize frame w h)
        (.setVisible frame true))))))

(defn write-bufferedimage
  "Write a BufferedImage to the specified file as a JPEG."
  [bi file-name]
  (javax.imageio.ImageIO/write bi "jpg" (java.io.File. file-name)))

(defn get-opencv-image
  "Load an image into an OpenCV matrix."
   ([path]
    (get-opencv-image path CvType/CV_32FC3))
   ([path type]
    (let [img (Imgcodecs/imread path)]
      (.convertTo img img type)
      img)))
