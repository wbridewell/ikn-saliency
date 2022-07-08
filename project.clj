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

(defproject ikn-saliency "1.0"
  :description "A Clojure implementation of the Itti, Koch, and Niebur approach to visual saliency."
  :url "http://example.com/FIXME"
  :license {:name "To Be Determined"
            :url "To Be Determined"}
  ;; insert location of your OpenCV 3.X java directory here.
  ;; the default location for Homebrew on MacOS X is included as an example.
  :jvm-opts ["-Djava.library.path=/usr/lib:/usr/local/lib:/usr/local/opt/opencv3/share/OpenCV/java/"]
  ;; NOTE: Version 4.X of OpenCV and 1.10 of Clojure should be substitutable
  ;; without any issues as long as the dependencies are installed appropriately.
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [local/opencv "3.2.0"]
                 [local/opencv-native "3.2.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]]
  ;; necessary for OpenCV to work.
  :injections [(clojure.lang.RT/loadLibrary org.opencv.core.Core/NATIVE_LIBRARY_NAME)]
  ;; jars for OpenCV are expected to be installed in a local repository called repo.
  :repositories { "project" {:url "file:repo" :update :always}})
