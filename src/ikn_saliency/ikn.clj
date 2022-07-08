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

(ns ikn-saliency.ikn
  ^{:doc "An implementation of Itti, Koch, and Niebur's approach to visual saliency."}
  (:use clojure.math.numeric-tower)
  (:import
    java.awt.image.BufferedImage
    [org.opencv.core Core Size Mat Point CvType Scalar]
    org.opencv.imgproc.Imgproc
    org.opencv.imgcodecs.Imgcodecs))

;; NOTE: original image must be at least 256x256 to support the construction of
;; the Gaussian pyramids. the top level saliency function checks for this
;; constraint and returns nil if it is unmet.

;;;;;;;;;;;;;;;;;
;; This file contains a stand-alone implementation of the algorithm for visual
;; saliency originally described in
;;
;; Itti, L., Koch, C., & Niebur, E. (1998). A model of
;; saliency-based visual attention for rapid scene analysis.
;; IEEE Transactions on Pattern Analysis and Machine Intelligence,
;; 20, 1254--1259.
;;
;; PAMI
;; http://ilab.usc.edu/publications/doc/Itti_etal98pami.pdf
;;
;; Some modifications to the algorithm are reported in
;;
;; Itti, L., Dhavale, N., & Pighin, F. (2003). Realistic avatar
;; eye and head animation using a neurobiological model of visual
;; attention. In Proceedings of SPIE, 5200, 64--78.
;;
;; SPIE
;; http://ilab.usc.edu/publications/doc/Itti_etal03spienn.pdf
;;
;; Additional reference is made to the Matlab version of the
;; algorithm as available in the Saliency Toolbox.
;;
;; MLSAL
;; http://www.saliencytoolbox.net
;;
;; Further reference is made to the C++ code from the Itti
;; lab, and specifically to the INVT/simple-saliency.C source
;; along with other required headers and source files.
;;
;; CPSAL
;; http://ilab.usc.edu/toolkit/home.shtml
;;
;; Note that substantial discrepancies exist between the
;; theoretical description of the algorithm and its implementation
;; in C++, Matlab, or here in Clojure. These differences are
;; reflected in the final saliency map and are occasionally reported
;; in the comments.
;;
;; Use of this code requires OpenCV.
;;
;; Unless otherwise noted, all operations assume that the matrices
;; are in CV_32FC1 format (one channel of 32-bit floats).
;;
;; For an introduction to image filtering and Gabor filters in
;; particular, see this resource from Patrick Fuller.
;; http://patrick-fuller.com/gabor-filter-image-processing-for-scientists-and-engineers-part-6/
;;
;; For an interactive website that lets you try different
;; parameterizations of Gabor filters and other image processing
;; routines, see this resource from Nicolai Petkov's laboratory at
;; the University of Gruningen.
;; http://matlabserver.cs.rug.nl
;;;;;;;;;;;;;;;;;;;




;;;;;
;; Border behavior when applying kernels is left unspecified in PAMI,
;; and I haven't bothered to track it down in MLSAL or CPSAL. I have
;; chosen to always use a reflected border, but other options are
;; available through OpenCV.
;;;;;
(def +border-behavior+ Core/BORDER_REFLECT_101)


;;;;;
;; Images are resized when different levels of the Gaussian pyramid
;; are combined using the center-surround operator. PAMI leaves the
;; form of interpolation unspecified. I have chosen to use either
;; nearest-neighbor or bilinear interpolation. OpenCV supports
;; other options as well.
;;;;;
(def +resize-interpolation+ Imgproc/INTER_LINEAR)
;(def +resize-interpolation+ Imgproc/INTER_NEAREST)

;;;;;
;; After convolution, the edges of the matrix will be disproportionately
;; salient. Setting a minimum number of pixels for edge attenuation above
;; zero ensures that the saliency at edges will always be reduced by
;; some degrees.
;;;;;
(def +minimum-attenuation+ 10)

;;;;;
;; Matrix operations
;;
;; Occasionally, it's useful to have a functional version of
;; a matrix operation. Those included here by no means cover
;; all the operations supported by OpenCV.
;;;;;

;; Oddly, Java OpenCV doesn't have an abs function, so this is what we do instead.
(defn matrix-abs
  "Generates the absolute value of an OpenCV matrix."
  [mtx]
  (let [new-mtx (Mat.)]
    (Core/absdiff mtx (Scalar/all 0) new-mtx)
    new-mtx))

(defn matrix-absdiff
  "Generates the absolute difference of two OpenCV matrices."
  [mtx1 mtx2]
  (let [dst (Mat.)]
    (Core/absdiff mtx1 mtx2 dst)
    dst))

(defn matrix-subtract
  "Generates the difference of two OpenCV matrices."
  [mtx1 mtx2]
  (let [dst (Mat.)]
    (Core/subtract mtx1 mtx2 dst)
    dst))

(defn matrix-divide
  "Generates the quotient of two OpenCV matrices."
  [mtx1 mtx2]
  (let [dst (Mat.)]
    (Core/divide mtx1 mtx2 dst)
    dst))

(defn matrix-multiply
  "Generates the product of two OpenCV matrices."
  [mtx1 mtx2]
  (let [dst (Mat.)]
    (Core/multiply mtx1 mtx2 dst)
    dst))

(defn matrix-add
  "Generates the sum of two OpenCV matrices."
  [mtx1 mtx2]
  (let [dst (Mat.)]
    (Core/add mtx1 mtx2 dst)
    dst))

(defn matrix-clamp-min
  "Generates a matrix with negative values below a threshold
  (assumed 0) clamped to 0."
  ([mtx]
   (matrix-clamp-min mtx 0.0))
  ([mtx threshold]
   (let [dst (Mat.)]
     (Imgproc/threshold mtx dst threshold 0.0 Imgproc/THRESH_TOZERO)
     dst)))

(defn matrix-clamp-max
  "Generates a matrix with values above threshold (default 255)
  clamped to the threshold."
  ([mtx]
   (matrix-clamp-max mtx 255.0))
  ([mtx threshold]
   (let [dst (Mat.)]
     (Imgproc/threshold mtx dst threshold 0.0 Imgproc/THRESH_TRUNC)
     dst)))

(defn clamp-color
  "Generates a matrix clamped to [0, 255]."
  [cmtx]
  (matrix-clamp-max (matrix-clamp-min cmtx 0.0) 255.0))

(defn normalize
  "Generates a matrix normalized between 0 and maximum."
  [mtx maximum]
  (let [dst (Mat.)]
    (Core/normalize mtx dst 0 maximum Core/NORM_MINMAX -1)
    dst))

(defn create-matrices
  "Generates a sequence of n OpenCV matrices."
  [n]
  (doall (pmap (fn [_] (Mat.)) (range n))))

(defn resize
  "Generates a matrix like mtx, but having the given size, and using
  the specified interpolation scheme."
  ([mtx size]
   (let [dst (Mat.)]
      (Imgproc/resize mtx dst size 0 0 +resize-interpolation+)
      dst))
  ([mtx size interpolation]
   (let [dst (Mat.)]
     (Imgproc/resize mtx dst size 0 0 interpolation)
     dst)))

;;;;;
;; Shift an image
;;
;; Movement based saliency maps compare an image to bit shifted versions.
;; There is no general operation for shifting images in OpenCV, but you can
;; copy pixels from one matrix to another. This function implements that to
;; shift an image up, down, left, or right a specified number of pixels.
;; Note that you can shift horizontally and vertically at the same time.
;;
;; shift left = -h-shift, right = +h-shift, up = -v-shift, down = +v-shift
;;
;; returns a blank image if the shifts are larger than the image dimensions.
(defn shift-image [img h-shift v-shift]
  (let [dst (Mat/zeros (.size img) (.type img))]
    (if (and (<= (abs v-shift) (.height img))
             (<= (abs h-shift) (.width img)))
      (.copyTo (.submat img
                 (if (>= v-shift 0) 0 (abs v-shift))
                 (if (>= v-shift 0) (- (.height img) v-shift) (.height img))
                 (if (>= h-shift 0) 0 (abs h-shift))
                 (if (>= h-shift 0) (- (.width img) h-shift) (.width img)))
               (.submat dst
                 (if (>= v-shift 0) v-shift 0)
                 (if (>= v-shift 0) (.height dst) (+ (.height dst) v-shift))
                 (if (>= h-shift 0) h-shift 0)
                 (if (>= h-shift 0) (.width dst) (+ (.width dst) h-shift)))))
    dst))

;;;;;
;; Edge attenuation
;;
;; Although not discussed in PAMI, both CPSAL and MLSAL attenuate
;; values along the edges of a matrix several times throughout
;; the algorithm. This can protect against artificially salient
;; regions that may appear near the edges and in the corners.
;;;;;
(defn lower-bound-attenuation
  "Ensures that the requested attenuation size is appropriate for the image."
  [width height atsize]
  (max +minimum-attenuation+ (if (> (* 2 atsize) (min width height))
                              (/ (min width height) 2)
                              (int atsize))))

;; TODO: Do we really need to turn this into a Java array and
;; process the elements one-by-one? That slows things down.
(defn attenuation-matrix
  "Generates a matrix useful for edge attenuation where the edges are of the requested width."
  [mtx rsize]
  (let [asz (lower-bound-attenuation (.width mtx) (.height mtx) rsize)]
    (if (< (int asz) 1)
      (Mat/ones (.size mtx) CvType/CV_32FC1)
      (let [width (.width mtx)
            height (.height mtx)
            jfarr (float-array (* width height) 1.0)
            increment (/ 1.0 (+ 1.0 asz))]
        ;; attenuate the top edge
        (loop [y 0
               coefficient increment]
          (when (< y asz)
            (dotimes [x width]
              (let [idx (+ (* y width) x)]
                (aset ^floats jfarr idx (* (aget ^floats jfarr idx) coefficient))))
            (recur (inc y) (+ coefficient increment))))

        ;; attenuate the bottom edge
        (loop [y (- height asz)
               coefficient (* increment asz)]
          (when (< y height)
            (dotimes [x width]
              (let [idx (+ (* y width) x)]
                (aset ^floats jfarr idx (* (aget ^floats jfarr idx) coefficient))))
            (recur (inc y) (- coefficient increment))))

        ;; attenuate the left and right edges
        ;; note that corners will be doubly attenuated on purpose
        (dotimes [y height]
          (loop [x 0
                 coefficient increment]
            (when (< x asz)
              (let [idxnear (+ (* y width) x)
                    idxfar (- (* (inc y) width) (inc x))]
                (aset ^floats jfarr idxnear (* (aget ^floats jfarr idxnear) coefficient))
                (aset ^floats jfarr idxfar (* (aget ^floats jfarr idxfar) coefficient)))
              (recur (inc x) (+ coefficient increment)))))
        ;; put the Java array into an OpenCV matrix
        (let [amtx (Mat. (.size mtx) CvType/CV_32FC1)]
          (.put amtx 0 0 jfarr)
          amtx)))))

(defn attenuate-borders
  "Attenuate the edges of the matrix using the given edge width."
  ([mtx]
   ;; This width definition 1/20 is commonly used in CPSAL
   (attenuate-borders mtx (/ (max (.width mtx) (.height mtx)) 20)))
  ([mtx size]
   (let [atmtx (attenuation-matrix mtx size)]
     (matrix-multiply mtx atmtx))))

;;;;;
;; Separating the colors into basic channels
;;
;; Although seemingly trivial, PAMI defines the basic red, green,
;; and blue color channels as more than the pixel values given
;; when reading an image. For the most part, red, green, blue,
;; and yellow are computed as defined in PAMI. CPSAL alters the
;; computation of yellow "to compensate for its attenuation."
;;
;; The function split-colors provides the data structure used
;; to compute R, B, G, and Y. This is a map where the matrices
;; are associated with keywords :red, :green, and :blue with
;; the obvious interpretations.
;;;;;
(defn split-colors
  "Given an OpenCV BGR matrix, returns a map that contains
   separate matrices for each color."
  [mtx]
  (let [cs (java.util.ArrayList.)]
    (Core/split mtx cs)
     ;; OpenCV stores images in BGR format.
    (zipmap '(:blue :green :red) (vec cs))))

(defn intensity-channel
  "Generate the intensity channel from a color map: (/ (+ b g r) 3)."
  [colors]
  (let [dst (Mat.)]
    (Core/add (:red colors) (:blue colors) dst)
    (Core/add dst (:green colors) dst)
    (Core/divide dst (Scalar/all 3) dst)
    dst))

(defn red-channel
  "Generate the red channel from a color map: (- r (/ (+ b g) 2)))."
  [colors]
  (let [dst (Mat.)]
    (Core/add (:green colors) (:blue colors) dst)
    (Core/divide dst (Scalar/all 2) dst)
    (Core/subtract (:red colors) dst dst)
    dst))

(defn green-channel
  "Generate the green channel from a color map: (- g (/ (+ r b) 2)))."
  [colors]
  (let [dst (Mat.)]
    (Core/add (:red colors) (:blue colors) dst)
    (Core/divide dst (Scalar/all 2) dst)
    (Core/subtract (:green colors) dst dst)
    dst))

(defn blue-channel
  "Generate the blue channel from a color map: (- b (/ (+ r g) 2)))."
  [colors]
  (let [dst (Mat.)]
    (Core/add (:red colors) (:green colors) dst)
    (Core/divide dst (Scalar/all 2) dst)
    (Core/subtract (:blue colors) dst dst)
    dst))

(defn yellow-channel-pami
  "Generate the yellow channel from a color map as described in PAMI:
   (- (/ (+ r g) 2)
      (/ (abs (- r g)) 2)
      b))."
  [colors]
  (let [tmp (Mat.)
        dst (Mat.)]
    (Core/add (:red colors) (:green colors) dst)
    (Core/absdiff (:red colors) (:green colors) tmp)
    (Core/divide dst (Scalar/all 2) dst)
    (Core/divide tmp (Scalar/all 2) tmp)
    (Core/subtract dst tmp dst)
    (Core/subtract dst (:blue colors) dst)
    dst))

;; This is the version implemented in the file ColorOps.C
(defn yellow-channel-intv
  "Generate the yellow channel from a color map as described in CPSAL:
   (* 2 (- (/ (+ r g) 2)
           (abs (- r g))
           b))."
  [colors]
  ;; yellow is doubled "to compensate for its previous attentuation"
  ;; from normalizing by the luminance (intensity).
  (let [tmp (Mat.)
        dst (Mat.)]
    (Core/add (:red colors) (:green colors) dst)
    (Core/absdiff (:red colors) (:green colors) tmp)
    (Core/divide dst (Scalar/all 2) dst)
    (Core/subtract dst tmp dst)
    (Core/subtract dst (:blue colors) dst)
    (Core/multiply dst (Scalar/all 2.0) dst)
    dst))

;; Currently uses the CPSAL version of yellow.
;; TODO: create a way to ease switching between different alternatives at the
;; level of CPSAL, PAMI, etc.
(defn yellow-channel
  "Generate the yellow channel from a color map."
  [colors]
  (yellow-channel-intv colors))

(defn intensity-threshold
  "Generates a matrix where intensity values below 25.5 are set to 0.0
   and remaining intensity values are divided by 255."
  [imtx]
  ;; although PAMI ambiguously suggests that imax = global max, it
  ;; really equals the maximum value of the range for the colors.
  ;; so it's a constant 255 in this case.
  (let [imax 255.0 ;; (.maxVal (Core/minMaxLoc imtx))
        tmtx (Mat.)]
    (Imgproc/threshold imtx tmtx (/ imax 10.0) 0 Imgproc/THRESH_TOZERO)
    ;; getRGBY in CPSAL also divides the intensity by 255
    ;; before dividing each base color to it. Who knew?
    (Core/divide tmtx (Scalar/all 255) tmtx)
    tmtx))

(defn normalize-by-intensity
  "Generates a new color map where the basic colors are
   normalized by intensity."
  [colors intensity]
  (let [dst (Mat.)
        threshold (intensity-threshold intensity)]
    {:red (matrix-divide (:red colors) threshold)
     :green (matrix-divide (:green colors) threshold)
     :blue (matrix-divide (:blue colors) threshold)}))

(defn color-channels
  "Generates a map of channel names and their corresponding
  color channels from an OpenCV CV_32FC3 image. Channel names
  are :red, :green, :blue, :yellow."
  [img intensity-channel]
  (let [colors (split-colors img)
        norm-colors (normalize-by-intensity colors intensity-channel)]
    {:red (clamp-color (red-channel norm-colors))
     :green (clamp-color (green-channel norm-colors))
     :blue (clamp-color (blue-channel norm-colors))
     :yellow (clamp-color (yellow-channel norm-colors))}))


(defn flicker-channel [previous-intensity current-intensity]
  {:flicker (matrix-absdiff previous-intensity current-intensity)})

;;;;;
;; Gabor filtering
;;
;; A Gabor filter combines a Gaussian filter with a Fourier transform,
;; which means that each one has several parameters. The filter is
;; often used for edge detection, and in visual saliency, it provides
;; information on edge orientations, ultimately providing the
;; orientation map.
;;
;; PAMI does not specify any particular parameters apart from the four
;; angles (theta) used to generate the basic orientation maps. An
;; examination of MLSAL and CPSAL shows that two phases (0 and 90
;; degrees) are merged for each angle. The parameters required by
;; OpenCV differ from the set used in MLSAL, and CPSAL carries out
;; Gabor filtering using a different strategy. The parameters used in
;; the current code are taken from neuroscientifically plausible
;; ranges.
;;
;; Since the filter is fully defined by its parameters, this is a good
;; place for memoization. If the system using this code will process
;; multiple images before exiting, then recalculating the filters is
;; nonsensical.
;;;;;

;; In the Fuller tutorial, sigma is set to
;; [wavelength * 1/PI * sqrt(log(2)/2) * (pow(2,octave)+1)/(pow(2,octave)-1)],
;; which relates the Gaussian kernel's sigma to the wavelength of the
;; Fourier kernel. In image processing, 1 is a common value for octave.
(defn calculate-sigma
  "Computes the value of the sigma parameter for a Gabor filter
   from a wavelength and an octave."
  [wavelength octave]
  (* wavelength (/ 1 Math/PI)
     (Math/sqrt (/ (Math/log 2) 2))
     (/ (+ (Math/pow 2 octave) 1)
        (- (Math/pow 2 octave) 1))))


;; This calculation also comes from the Fuller tutorial. In MLSAL,
;; size is a parameter and is set to 13 with no further comment.
(defn calculate-size
  "Computes the size of a Gabor kernel from the sigma parameter."
  [sigma]
  (let [x (Math/ceil (Math/sqrt (* -2 sigma sigma (Math/log 0.005))))]
    (+ (* 2 x)
       (if (= (mod x 2) 1) 1 0))))


;; The parameters here are taken from plausible ranges provided from
;; neuroscientific research. A quick description of these ranges is
;; available at the website for Nicolai Petkov's laboratory.
;; http://www.cs.rug.nl/~imaging/simplecell.html
(defn gabor-kernel
  "Generates a Gabor kernel of the given angle and phase
   using default parameters for the other parameters."
  [angle phase]
  ;; the Matlab code creates filters at phases 0 and 90.
  (let [octave 1.4 ;; the bandwidth is specified in octaves
                   ;; (1.4 is the weighted average for macaques)
        lambda 3 ;; Octave, lambda, and sigma are interdependent and
                 ;; also influence the size of the window.
                 ;; With the current parameterization, a lambda of
                 ;; 3 yields a size of 10 pixels for the filter.
        sigma (calculate-sigma lambda octave)
        size (calculate-size sigma)
        ;; The value of theta is equal to the angle, here converted to radians.
        theta (* (/ Math/PI 180) angle) ; convert to radians
        gamma 1.0 ;; between 0.2 and 1.0 for simple cells
        ;; The value of psi is equal to the phase, here converted to radians.
        psi (* (/ Math/PI 180) phase)
        kernel (Imgproc/getGaborKernel (Size. size size)
                                       sigma theta lambda gamma psi CvType/CV_32F)]
    ;; MLSAL normalizes the kernel before returning it, but that step
    ;; should be unnecessary.
    kernel))


;; This is a basic implementation of linspace from Matlab and Numpy
;; http://www.mathworks.com/help/matlab/ref/linspace.html
;; "Generate linearly spaced vectors"
;;
;; (linspace a b n) "generates a vector of n points linearly spaced
;; between and including a and b. For n = 1, linspace returns b."
(defn linspace
  "Generates a vector of n points linearly spaced between and
   including min and max. For n = 1, linspace returns max."
  [min max n]
  (if (= n 1)
    (vector max)
    (let [step (/ (- max min) (dec n))]
      (into [] (range min (+ max step) step)))))

;; This is the function to call for creating Gabor kernels. Consider
;; adding one more layer on top of this that memoizes the kernels
;; with the default number of angles from PAMI.
;; returns a sequence of Gabor kernels of length num-angles.
;; each item is a tuple of filters at phases 0 and 90.
;; this follows the strategy of the saliency toolbox code.
;; The original paper suggests num-angles = 4 [0, 45, 90, 135]
(defn get-gabor-kernels
  "Generates a sequence of Gabor-kernel tuples containing kernels for
   phases 0 and 90. The sequence contains num-angles tuples for angles
   spaced equally in [0, 180)."
  [num-angles]
  (let [angs (linspace 0 (- 180 (/ 180 num-angles)) num-angles)]
    (pmap (fn [angle] [(gabor-kernel angle 0) (gabor-kernel angle 90)]) angs)))

;; Phases is a tuple containing matrices filtered through phase 0 and
;; phase 90 Gabor kernels with other parameters held constant. This
;; function will return a Mat that contains the L1 Norm of the elements
;; in the matrices.
(defn combine-phases
  "Generates the L1 norm of the elements in the given matrix tuple."
  [phases]
 (matrix-add (matrix-abs (first phases)) (matrix-abs (second phases))))

(defn apply-gabor
  "Generates a sequence of orientation channels produced by filtering
   the given image through the sequence of kernel tuples."
  [kernels imap]
  (doall (pmap (fn [ks]
                (let [k0 (Mat.)
                      k90 (Mat.)]
                  (Imgproc/filter2D imap k0 -1 (first ks) (Point. -1 -1) 0 +border-behavior+)
                  (Imgproc/filter2D imap k90 -1 (second ks) (Point. -1 -1) 0 +border-behavior+)
                  ;; CPSAL uses a slight attenuation of the image borders after filtering
                 ; (combine-phases [k0 k90])))
                  (combine-phases [(attenuate-borders k0 5) (attenuate-borders k90 5)])))
              kernels)))


;; Produces all the channels used in PAMI.
(defn create-basic-channels
  "Generates a map of channel names and their corresponding
  color channels from an OpenCV CV_32FC3 image. Channel names
  are :red, :green, :blue, :yellow, :intensity, :flicker, and
  :orientations, the last of which includes a sequence of
  channels for angles 0, 45, 90, and 135 degrees."
  [current-img & {:keys [previous-img features]}]
  (let [ic (intensity-channel (split-colors current-img))]
    (merge {}
           (when (contains? features :color) (color-channels current-img ic))
           (when (contains? features :intensity) {:intensity ic})
           (when (contains? features :orientations)
             {:orientations (apply-gabor (get-gabor-kernels 4) ic)})
           (when (and previous-img (contains? features :flicker))
             {:flicker (matrix-absdiff (intensity-channel (split-colors previous-img)) ic)}))))

;;;;;
;; Resizing images using Gaussian pyramids.
;;
;; Computing center-surround subtracts a coarsely grained
;; version of a channel from a more finely grained one. The
;; different granularities are computed by downsampling an
;; image using a Gaussian kernel. In PAMI and CPSAL, images
;; are downsampled by powers of 2 from 2^1 to 2^8. Center
;; surround is computed using images from 2^2 through 2^8.
;; In papers and code, the exponents are frequently called
;; octaves, not to be confused with the octave parameter
;; for the Gabor filter.
;;;;;

(defn create-interpolated-gaussians
  "Generates a vector containing downsampled images at octaves
   from 0 to max-octave."
  ([img max-octave]
   (let [sizes (pmap #(Size. (round (/ (.width img)  (expt 2 %1)))
                            (round (/ (.height img) (expt 2 %1))))
                    (range 1 (inc max-octave)))]
    (create-interpolated-gaussians img sizes '[])))
  ([img sizes acc]
     ;; This is a helper function that shouldn't be called
     ;; directly. The size handed to pyrDown must be half
     ;; the size of the image it's given.
   (if (empty? sizes)
     acc
     (let [mat (Mat.)]
         ;; The kernel size is 5x5 as in CPSAL.
         ;; note that orientation kernels in CPSAL are 9x9, but here they are 5x5.
       (Imgproc/pyrDown img mat (first sizes) +border-behavior+)
       (recur mat (rest sizes) (conj acc mat))))))

;; Depending on the structure of the input map,
;;
;; * Keeps the red, green, blue, and yellow channels separate until
;;   later stages of the algorithm. (PAMI)
;; * Implements red-green calculations as used in CPSAL where
;;   red-green = R-G and blue-yellow = B-Y.
(defn create-gaussian-pyramids
  "Generates a map of the Gaussian pyramids used for center-surround calculations."
  [channel-map]
  (let [top-octave 8]
    (merge {}
           (when (contains? channel-map :intensity)
             {:intensity (create-interpolated-gaussians (:intensity channel-map) top-octave)})
           (when (contains? channel-map :red) ;; color is all or none for red, green, blue, yellow basic features (PAMI)
             {:red (create-interpolated-gaussians (:red channel-map) top-octave)
              :blue (create-interpolated-gaussians (:blue channel-map) top-octave)
              :green (create-interpolated-gaussians (:green channel-map) top-octave)
              :yellow (create-interpolated-gaussians (:yellow channel-map) top-octave)})
           (when (contains? channel-map :red-green) ;; color is all or none for red-green, blue-yellow (CPSAL)
             {:red-green (create-interpolated-gaussians (:red-green channel-map) top-octave)
              :blue-yellow (create-interpolated-gaussians (:blue-yellow channel-map) top-octave)})
           (when (contains? channel-map :orientations)
             {:orientations (pmap #(create-interpolated-gaussians %1 top-octave) (:orientations channel-map))})
           (when (contains? channel-map :flicker) {:flicker (create-interpolated-gaussians (:flicker channel-map) top-octave)}))))

;;;;;
;; Applying center-surround
;;
;;
;; PAMI reports color center-surround as
;;
;;   RG(c,s) = |R(c)-G(c) (-) G(s)-R(s)|
;;
;; where (-) is the center-surround operator. The equation for
;; BY is analogous. In contrast, SPIE reports the equation as
;;
;;   RG(c,s) = |R(c)-G(c) (-) R(s)-G(s)|
;;
;; reversing the order of the colors in the surround condition.
;; In this code, we opt for the SPIE version, which is also
;; reflected in CPSAL.
;;;;;

(defn bsc-cs
  "Generates the center-surround matrix."
  [center surround]
  (attenuate-borders (matrix-absdiff center (resize surround (.size center)))))

(defn basic-center-surround
  "Generates a nested map of center-surround results where the first-level
   map is keyed by the center indices (e.g., :2) and the second-level maps
   are keyed by the surround indices."
  [pyr]
  {:2 {:5 (bsc-cs (get pyr 1) (get pyr 4))
       :6 (bsc-cs (get pyr 1) (get pyr 5))}
   :3 {:6 (bsc-cs (get pyr 2) (get pyr 5))
       :7 (bsc-cs (get pyr 2) (get pyr 6))}
   :4 {:7 (bsc-cs (get pyr 3) (get pyr 6))
       :8 (bsc-cs (get pyr 3) (get pyr 7))}})


;; In CPSAL, we see that RG is first calculated as R-G before
;; the gaussian pyramids are applied, and likewise for BY.
;; Then, each of RG and BY is treated the same as the
;; intensity channel.
(defn center-surround
  "Generates a map of center-surround results."
  [pyramids]
  (merge {}
         (when (contains? pyramids :intensity)
           {:intensity (basic-center-surround (:intensity pyramids))})
         ;; treating the color channels as in Itti, Dhavale, & Pighin, 2003 (PAMI).
         (when (contains? pyramids :red) ;; color of red, blue, green, yellow is all or none
           {:red-green (basic-center-surround pyramids
                                              (pmap #(matrix-subtract %1 %2)
                                                    (:red pyramids)
                                                    (:green pyramids)))
            :blue-yellow (basic-center-surround pyramids
                                                (pmap #(matrix-subtract %1 %2)
                                                      (:blue pyramids)
                                                      (:yellow pyramids)))})
         ;; treating the color channels as in the Itti lab code (CPSAL).
         (when (contains? pyramids :red-green) ;; color of red-green, blue-yellow is all or none
           {:red-green (basic-center-surround (:red-green pyramids))
            :blue-yellow (basic-center-surround (:blue-yellow pyramids))})
         (when (contains? pyramids :orientations)
           {:orientations (vec (pmap #(basic-center-surround %1) (:orientations pyramids)))})
         (when (contains? pyramids :flicker)
           {:flicker (basic-center-surround (:flicker pyramids))})))

;;;;;
;; Generating the feature maps
;;
;; The algorithm for visual saliency generates several feature maps that
;; are eventually combined into conspicuity maps according to their feature
;; types. This set of functions generates feature maps either according to
;; SPIE or according to CPSAL.
;;;;;

(defn create-feature-maps-spie
  "Generates the 42 static feature maps and 4 flicker maps of an image as
  described by Itti, Dhavale, & Pighin, 2003."
  [current-img & {:keys [previous-img features]}]
  (center-surround
    (create-gaussian-pyramids
      (create-basic-channels current-img :previous-img previous-img :features features))))

(defn proto-feature-maps-cpsal [bc]
  (merge {}
         (when (contains? bc :intensity) {:intensity (:intensity bc)})
         (when (contains? bc :red) ;; color is all or none for basic features
           ;; note that there may be negative values in the color channels
           {:red-green (matrix-subtract (:red bc) (:green bc))
            :blue-yellow (matrix-subtract (:blue bc) (:yellow bc))})
         (when (contains? bc :orientations) {:orientations (:orientations bc)})
         (when (contains? bc :flicker) {:flicker (:flicker bc)})))

(defn create-feature-maps-cpsal
  "Generates the 42 static feature maps of an image and the 4 flicker maps
  as implemented in simple-saliency.C in the Itti lab code."
  [current-img & {:keys [previous-img features]}]
  (center-surround
   (create-gaussian-pyramids
    (proto-feature-maps-cpsal (create-basic-channels current-img :previous-img previous-img :features features)))))

;;;;;
;; Working with motion saliency
;;
;; after creating feature maps, add the motion feature maps if desired.
;;

;; Takes an image pyramid and shifts each level horizontally and vertically by
;; the specified number of pixels
(defn shift-pyramid [pyramid horizontal vertical]
  (pmap #(shift-image %1 horizontal vertical) pyramid))

;; unshifted current * shifted past - unshifted past * shifted current
(defn compute-reichardt-map [c sc p sp]
  (matrix-absdiff (matrix-multiply c sp) (matrix-multiply p sc)))

(defn compute-reichardt-pyramid [c-pyr sc-pyr p-pyr sp-pyr]
  (pmap #(compute-reichardt-map %1 %2 %3 %4) c-pyr sc-pyr p-pyr sp-pyr))

(defn motion-features [current-pyr previous-pyr]
  (let [threshold 3.0
        right (compute-reichardt-pyramid current-pyr (shift-pyramid current-pyr 1 0)
                                         previous-pyr (shift-pyramid previous-pyr 1 0))
        left  (compute-reichardt-pyramid current-pyr (shift-pyramid current-pyr -1 0)
                                         previous-pyr (shift-pyramid previous-pyr -1 0))
        up    (compute-reichardt-pyramid current-pyr (shift-pyramid current-pyr 0 -1)
                                         previous-pyr (shift-pyramid previous-pyr 0 -1))
        down  (compute-reichardt-pyramid current-pyr (shift-pyramid current-pyr 0 1)
                                         previous-pyr (shift-pyramid previous-pyr 0 1))]
    ;; remove small motion
    [ (vec (pmap #(matrix-clamp-min %1 threshold) right))
      (vec (pmap #(matrix-clamp-min %1 threshold) left))
      (vec (pmap #(matrix-clamp-min %1 threshold) up))
      (vec (pmap #(matrix-clamp-min %1 threshold) down))]))

(defn create-motion-feature-map [current-img previous-img]
  (let [top-octave 8]
    (vec (pmap #(basic-center-surround %1)
              (motion-features
                (create-interpolated-gaussians (intensity-channel (split-colors current-img)) top-octave)
                (create-interpolated-gaussians (intensity-channel (split-colors previous-img)) top-octave))))))




(defn create-feature-maps
  "Generates feature maps to  be combined into conspicuity maps. The maps
  are organized by feature (e.g., :intensity, :orientations) and these
  each contain center-surround results organized as nested maps keyed first
  by the center index and second by the surround index."
  [current-img & {:keys [previous-img features]}]
  (merge (create-feature-maps-cpsal current-img :previous-img previous-img :features features)
         (when (and previous-img (contains? features :motions))
           {:motions (create-motion-feature-map current-img previous-img)})))

;;;;;
;; Map normalization
;;
;; In PAMI, the authors describe a normalization operator N(.) that is supposed
;; to dampen background noise in the image, heightening the signal of the global
;; maximum if its value is far enough away from local maxima. In that algorithm,
;; local maxima are defined as being points that are greater than their 8
;; neighboring points. In CPSAL, this appears to be simplified to remove the
;; four diagonals.
;;
;; The steps in this algorithm are to
;; (1) normalize the map between 0 and a global maximum M (CPSAL uses 10.0).
;; (2) compute the average of all local maxima mbar.
;; (3) globally multiply the map from (1) by (M-mbar)^2.
;;
;; Later papers, including SPIE, use a difference of Gaussians approach to
;; normalization, which better dampens the background noise.
;;;;;

;; The dilation + compare approach to identifying local maxima is described
;; in response to a question at Stack Overflow.
;; http://stackoverflow.com/questions/10621312/opencv-filter-image-replace-kernel-with-local-maximum
(defn local-maxima-map
  "Generates a map that retains only the local maxima and clamps values
   below threshhold to zero."
  [fmap thresh]
  (let [kernel (Imgproc/getStructuringElement Imgproc/MORPH_RECT (Size. 3 3))
        maxed (Mat. (.width fmap) (.height fmap) (.type fmap))
        comp (Mat. (.width fmap) (.height fmap) (.type fmap))]
    (Imgproc/dilate fmap maxed kernel)
    (Core/compare fmap maxed comp Core/CMP_EQ)
    ;; compare gives us a CV_8UC1 map, so we need to convert it for later processing.
    (.convertTo comp comp CvType/CV_32FC1)
    (Core/multiply fmap comp maxed (/ 1 255.0))
    (matrix-clamp-min maxed thresh)))

(defn count-local-maxima
  "Computes the number of nonzero elements in a matrix."
  [max-map]
  (let [tmp (Mat.)
        maxes (Mat.)]
    ;; findNonZero only works on one type of matrix.
    ;; to deal with rounding, just set all non-zero values to 10
    (Imgproc/threshold max-map tmp 0 10 Imgproc/THRESH_BINARY)
    (.convertTo tmp tmp CvType/CV_8UC1)
    (Core/findNonZero tmp maxes)
    (.height maxes)))

(defn sum-local-maxima
  "Computes the sum of all values in a matrix."
  [max-map]
  ;; there's only one dimension
  (aget ^doubles (.val (Core/sumElems max-map)) 0))

;; fmap * (GlobalMaximum - mean(LocalMaxima))^2
(defn pami-norm-feature-map
  "Generates a matrix by multiplying the input by the normalization operator
  defined in Itti, Koch, & Niebur, 1998, with M = gmax (10 by default)."
  ([fmap]
   ;; in most cases, the Itti code uses 10.0
   ;; the idea is to remove amplitude differences across modalities.
   (pami-norm-feature-map fmap 10.0))
  ([fmap gmax]
   (let [dst (Mat.)
         nmap (normalize fmap gmax)
          ;; Itti et al., 1998 (PAMI) avoid values smaller than 1.
          ;; (min + (max - min)/10)
          ;; min is always 0 in practice
         lmap (local-maxima-map nmap (/ gmax 10.0))]
     (Core/multiply nmap
                    (Scalar/all (if (zero? (count-local-maxima lmap))
                                  0
                                  (expt (- gmax (/ (sum-local-maxima lmap)
                                                   (count-local-maxima lmap)))
                                        2)))
                    dst)
     dst)))

;;;;;
;; First-pass implementation of the difference of Gaussians normalization
;; function described in Itti & Koch, 2001 (Feature combination strategies
;; for saliency-based visual attention systems). The problem with this
;; approach is that the filtering function they use treats borders in a
;; special, nonstandard manner. Building a new filtering function may have
;; minimal payoff in comparison to the time to encode it in C++ as a part
;; of OpenCV. A Clojure version would likely be intolerably slow, but as
;; a sanity check, it may be in order.
;;;;;

(defn dog-norm-feature-map
  ([fmap]
   (dog-norm-feature-map fmap 10.0))
  ([fmap gmax]
   (let [nmap (normalize (matrix-clamp-min fmap) gmax)
         size (max (.width nmap) (.height nmap))
         maxhw (max 0 (dec (/ (min (.width nmap) (.height nmap)) 2)))
         ;; 2.0 and 25.0 are default numbers from INVT
         excitation-sigma (* size 0.02)
         inhibition-sigma (* size 0.25)
         gauss-excite (Imgproc/getGaussianKernel maxhw excitation-sigma CvType/CV_32F)
         gauss-inhibit (Imgproc/getGaussianKernel maxhw inhibition-sigma CvType/CV_32F)
         result (.clone fmap)
         excite (Mat.)
         inhibit (Mat.)
         ;; increasing the number of iterations can dramatically slow down processing since
         ;; this particular implementation of DoG is not particularly efficient
         ;;    5 is the default number in INVT code
         ;;    11 is used in the JEI paper
         iterations 3]
     (Core/multiply gauss-excite (Scalar/all (/ 0.5 (* excitation-sigma (sqrt (* 2 Math/PI))))) gauss-excite)
     (Core/multiply gauss-inhibit (Scalar/all (/ 1.5 (* inhibition-sigma (sqrt (* 2 Math/PI))))) gauss-inhibit)
     (dotimes [x iterations]
        (Imgproc/sepFilter2D result excite -1 gauss-excite gauss-excite
                             (Point. -1 -1) 0 Core/BORDER_REFLECT)
        (Imgproc/sepFilter2D result inhibit -1 gauss-inhibit gauss-inhibit
                             (Point. -1 -1) 0 Core/BORDER_REFLECT)
        (let [global-inhibition (Scalar/all (* 0.02 (.maxVal (Core/minMaxLoc result))))]
          (Core/add result (matrix-subtract excite inhibit) result)
          (Core/subtract result global-inhibition result)
          (Imgproc/threshold result result 0.0 0.0 Imgproc/THRESH_TOZERO)))
     result)))

(defn normalize-feature-map
  ([fmap] (normalize-feature-map fmap 1.0))
  ([fmap gmax] (dog-norm-feature-map fmap gmax)))

;; Takes something of the form {:2 {:5 ...} ...}
(defn normalized-maps
  "Generates a sequence of feature maps that are normalized versions
   of those in the center-surround maps for a single feature."
  [sfmaps]
  ;; take a collection of center surround feature maps for a single
  ;; feature, normalize them, and flatten them into a list.
  (pmap #'normalize-feature-map (flatten (pmap #'vals (vals sfmaps)))))

;;;;;
;; Creating conspicuity maps
;;
;; The phrase "circle plus sum" refers to the operator for adding features
;; to create conspicuity maps. This involves resizing each map to scale four
;; and adding them together.
;;;;;

(defn circle-plus-sum [mtxs]
  (let [target-size (Size. (apply min (pmap #(.width %1) mtxs))
                           (apply min (pmap #(.height %1) mtxs)))
        maps (pmap #(resize %1 target-size Imgproc/INTER_AREA) mtxs)
        dst (first maps)]
    (doseq [m (rest maps)]
      ;; CPSAL normalizes dst after every addition
      (Core/add dst m dst))
    dst))


(defn intensity-conspicuity
  "Generates the intensity conspicuity map from a map of feature maps."
  [fmaps]
  (circle-plus-sum (normalized-maps (:intensity fmaps))))

(defn flicker-conspicuity
  "Generates the flicker conspicuity map from a map of feature maps."
  [fmaps]
  (circle-plus-sum (normalized-maps (:flicker fmaps))))

(defn color-conspicuity
  "Generates the color conspicuity map from a map of feature maps."
  [fmaps]
  (circle-plus-sum
   (pmap #'matrix-add
        (normalized-maps (:red-green fmaps))
        (normalized-maps (:blue-yellow fmaps)))))

(defn orientation-conspicuity
  "Generates the orientation conspicuity map from a map of feature maps."
  [fmaps]
  ;; first, sum the maps for each of the orientations
  (let [mtxs (pmap #'circle-plus-sum (pmap #'normalized-maps (:orientations fmaps)))
        dst (normalize-feature-map (first mtxs))]
    ;; now, sum each of the orientation maps
    (doseq [m (rest mtxs)]
      (Core/add dst (normalize-feature-map m) dst))
    dst))

(defn motion-conspicuity
  "Generates the motion conspicuity map from a map of feature maps."
  [fmaps]
  ;; first, sum the maps for each of the orientations
  (let [mtxs (pmap #'circle-plus-sum (pmap #'normalized-maps (:motions fmaps)))
        dst (normalize-feature-map (first mtxs))]
    ;; now, sum each of the orientation maps
    (doseq [m (rest mtxs)]
      (Core/add dst (normalize-feature-map m) dst))
    dst))

(defn add-maps
  "Adds a sequence of matrices, ignoring nil entries."
  [mats]
  (let [m (remove nil? mats)
        dst (when (first m) (.clone (first m)))]
    (doseq [nxt (rest m)]
      (Core/add dst nxt dst))
    dst))

;; although not in PAMI, the saliency code weights each
;; channel by the number of feature maps that it contains.
(defn saliency-map
  "Generates the saliency map from a map of feature maps."
  [fmaps]
  (let [dst (Mat.)
        imap  (when (:intensity fmaps) (normalize-feature-map (intensity-conspicuity fmaps)))
        cmap  (when (:red-green fmaps) (normalize-feature-map (color-conspicuity fmaps)))
        omap  (when (:orientations fmaps) (normalize-feature-map (orientation-conspicuity fmaps)))
        flmap (when (:flicker fmaps) (normalize-feature-map (flicker-conspicuity fmaps)))
        mmap  (when (:motions fmaps) (normalize-feature-map (motion-conspicuity fmaps)))
        ;; there are 6 maps per feature c in [2,3,4] and d in [3,4]
        w (/ 1.0 6.0)]
    ;; weight each channel
    (when imap (Core/multiply imap (Scalar/all w) imap))
    (when cmap (Core/multiply cmap (Scalar/all (/ w 2.0)) cmap)) ;; by & rg
    (when omap (Core/multiply omap (Scalar/all (/ w 4.0)) omap)) ;; 0, 45, 90, 135
    (when flmap (Core/multiply flmap (Scalar/all w) flmap))
    (when mmap (Core/multiply mmap (Scalar/all (/ w 4.0)) mmap)) ;; up, down, left, right
    
    ;; PAMI divides by three instead of using the proportional weighting scheme.
    ;; This looks like a typographical error, so we don't even offer that as an option.
    ;;
    ;; the Itti code does a final N(.) pass after combining the channels
    {:saliency (normalize (normalize-feature-map (add-maps [imap cmap omap flmap mmap]) 10.0) 255)
     :intensity imap
     :color cmap
     :orientation omap
     :flicker flmap
     :motion mmap}))
