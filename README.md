# Computer vision
This repository contains 5 exercises for class Advanced Computer Vision Methods.

### Optical flow
In the first exercise our goal was to estimate the optical flow on consecutive images. 
We compared Lucas-Kanade and Horn-Schunck methods, first on rotated random noise images, and then on some real-world videos.
We also compared both methods with a combined method, where we initialize HS with output of LK.
Our main observations were that HS is better at recognizing the parts of image that don't move, and it overall has better 
performance, but it is also around 10 times slower, which has a big impact in real time video analysis.

### Mean-Shift tracking
In our second exercise we implemented our first simple tracker. It utilizes
Mean-shift algorithm for finding the region with the best
similarity between the region and a stored visual model of the
target object. The Mean-shift provides a robust deterministic
way of finding a mode of a probability density function.
We tested our tracker on [Vot14](https://www.votchallenge.net/vot2014/) sequences and proved good for sequences with slow 
movements, and failed on those where objects were quick.
Although it is clear there is a lot of room for improvements, the simple implementation proved promising and showed good 
performance on these sequences.

### Correlation filters
In this exercise we implemented correlation filter tracker. The idea behind this
approach is to find the filter that has a high response on the
object we are tracking, and low response on the background.
To speed up the tracking, we are calculating the response in
the Fourier domain, where correlation can be calculated as a
point wise product. We again evaluated the tracker on Vot14 sequences, using [Pytracking toolkit](https://github.com/alanlukezic/pytracking-toolkit-lite).
that allowed us to compare average overlap, number of failures and average speed of trackers with different parameters, 
to better understand their consequences. 

### Advanced tracking
Next we implemented a more advanced tracking
approach - combination of motion and visual model. We combined
a tracker that uses particle filter sequential optimization,
comparing histograms extracted at each particle, and Kalman
filter motion prediction. For motion prediction we derived and inspected different
models: random walk, nearly constant velocity and nearly
constant acceleration model. We again tested the tracker on Vot14 sequences with Pytracking toolkit.
This tracker proved better than both previous ones we implemented.
The biggest advantage could be the flexibility
between accuracy and speed. We can easily set the number of
particles so that it adjusts to our needs, possibly even automatically. Since we only tried
one, we could also combine it with different visual
models, to reach even better results.

### Long-term tracking
Up until now, all the trackers we implemented, needed manual
interaction to continue tracking after the target was lost. In
this exercise our goal was to find an approach that will detect when
the tracker loses the target, start searching for it, and perform
re-detection when it becomes visible again. To do this, we
modified an existing [SiamFC tracker](https://github.com/huanglianghua/siamfc-pytorch) 
into long-term tracker.
On targets that have a possibility to disappear,
long-term tracking improves performance, even with little
optimization. All our previous trackers had
good performance only because it was possible to manually
correct them when they lost the target. But this tracker has
the possibility to continue without intermission, which is a great
advantage.