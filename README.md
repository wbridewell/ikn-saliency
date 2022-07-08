# ikn-saliency

'IKN-Saliency' is an implmentation of Itti, Koch, and Niebur's approach to
computing visual saliency.

## Releases and Dependency Information

Latest stable release: 1.0.0

NOTE: Upon release, this section will also include information about where to access
the software.

### Dependencies and Compatibility

This library requires OpenCV with FFmpeg and Java bindings. Getting OpenCV to work with Clojure is described in [this tutorial].

[this tutorial]: https://docs.opencv.org/3.4/d7/d1e/tutorial_clojure_dev_intro.html

### Install OpenCV into a local Maven repository

To use OpenCV, you need JARs for both the OpenCV Java API and the native library.
These archives will need to be accessible in a Maven repository. The instructions
below assume an installation of OpenCV 3.2.0 on macOS on Intel processors, but 
should be straightforwardly adaptable to other versions and environments.

Copy the necessary files from their location into your project directory.
```
cp <location of OpenCV>/3.2.0/share/OpenCV/java/opencv-320.jar ./opencv.jar
cp <location of OpenCV>/3.2.0/share/OpenCV/java/libopencv_java320.dylib ./.
```

Create the JAR for the OpenCV native library.
```
mkdir -p native/macosx/x86_64
mv libopencv_java320.dylib native/macosx/x86_64
jar -cMf opencv-native.jar native
```

Deploy the JAR files using Maven.

```
mvn deploy:deploy-file -DgroupId=local -DartifactId=opencv -Dversion=3.2.0 -Dpackaging=jar -Dfile=opencv.jar -Durl=file:repo
```

```
mvn deploy:deploy-file -DgroupId=local -DartifactId=opencv-native -Dversion=3.2.0 -Dpackaging=jar -Dfile=opencv-native.jar -Durl=file:repo
```

Ensure that your project.clj file contains a reference to the OpenCV libraries under `:dependencies`.

```
[local/opencv "3.2.0"]
[local/opencv-native "3.2.0"]
```

## Introduction

This code is primarily based on the original algorithm described in

Itti, L., Koch, C., & Niebur, E. (1998). A model of
saliency-based visual attention for rapid scene analysis.
IEEE Transactions on Pattern Analysis and Machine Intelligence,
20, 1254--1259. [PAMI]

[PAMI]: http://ilab.usc.edu/publications/doc/Itti_etal98pami.pdf

Some modifications to their original algorithm are reported in

Itti, L., Dhavale, N., & Pighin, F. (2003). Realistic avatar
eye and head animation using a neurobiological model of visual
attention. In Proceedings of SPIE, 5200, 64--78. [SPIE]

[SPIE]:  http://ilab.usc.edu/publications/doc/Itti_etal03spienn.pdf

During implementation, additional reference was made to the Matlab version of the
algorithm as available in the [Saliency Toolbox].

[Saliency Toolbox]: http://www.saliencytoolbox.net

Further reference was made to the C++ code from the [Itti lab], and specifically
to the INVT/simple-saliency.C source along with other required headers and source files.

[Itti lab]: http://ilab.usc.edu/toolkit/home.shtml

Note that substantial discrepancies exist between the
theoretical description of the algorithm and its implementation
in C++, Matlab, or here in Clojure. These differences are
reflected in the final saliency map and are occasionally reported
in the comments.

## Example Usage

```clojure
(use 'ikn-saliency.core)
(require '[ikn-saliency.utils :as u])

(display-saliency-map "resources/pop-out-basic.png")

(def smap (saliency (u/get-opencv-image "resources/pop-out-basic.png")))
(u/display-image (u/mat-to-bufferedimage (:color smap)))
```

### Usage Notes
In addition to an OpenCV Mat containing an image, the function `ikn-saliency.core/saliency` takes a previous image and a set of features. The previous image is required for calculating the effects of motion and flicker and is useful for working with subsequent frames in a video. The available features include
* `:color`
* `:intensity`
* `:orientations`
* `:flicker`
* `:motions`

If no previous image is provided, `:flicker` and `:motions` will be ignored.

The function will return a map that includes a saliency map and conspicuity maps for each selected feature. The structure of this map is
```clojure
{:saliency     <OpenCV Mat>
 :color        <OpenCV Mat>
 :intensity    <OpenCV Mat>
 :orientation  <OpenCV Mat>
 :flicker      <OpenCV Mat>
 :motions      <OpenCV Mat>}
 ```
For features that are not calculated, the corresponding value will be `nil`.

## Resources

For an introduction to image filtering and Gabor filters in
particular, see the tutorial from [Patrick Fuller].

[Patrick Fuller]: https://patrickemmettfuller.com/gabor/


## Copyright and License

See the LICENSE file