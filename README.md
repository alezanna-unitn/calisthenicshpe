# calisthenicshpe
Zed &amp; MediaPipe python solutions for tracking Calisthenics static poses. Helpful to Judges during competitions.

Idea: Track 3 points on the athlete's body: shoulder, hip and ankle, connect them with a line and see if this line breaks, rises or lowers compared to how it should be performed, following the guidelines of the competition regulations in question.

Solution:

Input: Manually annotated video of poses (present in a folder).
Process:
Measure the difference between the coordinates of the points of the annotated video with the points tracked by the Zed and Mediapipe algorithms.
Verify that the algorithms are reliable for judging.
Calculate the angles between the points.
Measure the percentage of Malus that the athlete will receive in performing his skill:
0%,15%,25%,50%,100%
Output: 
- Mean Squared Error and Mean Per Joint Position Error to verify reliability of the algorithms
- Percentage of the malus according to the form of the pose
