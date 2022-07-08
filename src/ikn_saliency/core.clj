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

(ns ikn-saliency.core
  ^{:doc "An implementation of Itti, Koch, and Niebur's approach to visual saliency."}
  (:require [ikn-saliency.ikn :as ikn]
            [ikn-saliency.utils :as u])
  (:import org.opencv.core.Size))

;; NOTE: features refers to :orientations and :motions plural, but saliency
;; returns conspicuity maps with keys :orientation and :motion singular.
(defn saliency
  "Returns a map that includes OpenCV matrices corresponding to :saliency and
  the conspicuity maps for :color :intensity :orientation :flicker and :motion.
  A previous image is required to use :flicker or :motions."
  [image & {:keys [previous-img features]
            :or {features #{:color :intensity :orientations :flicker :motions}}}]
  (when (and (> (.width image) 255) (> (.height image) 255)))
  (u/apply-to-values (ikn/saliency-map (ikn/create-feature-maps image :previous-img previous-img :features features))
                     #(when (not (nil? %)) (ikn/resize % (Size. (.width image) (.height image))))))

(defn display-saliency-map
  "Displays the saliency map for an image at the specified path."
  [path]
  (u/display-image
   (u/mat-to-bufferedimage
    (:saliency (saliency (u/get-opencv-image path) :features #{:color :intensity :orientations})))))
