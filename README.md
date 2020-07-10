# Fall detection of a person in a video.

Fall detection is an important subject for hospitals: there are patience who should stay in bed without movements and who are too shy (or uncapable) to ask for a help.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=g-Hje8FJPAk
" target="_blank"><img src="https://sun6-16.userapi.com/HeU1wlantCQV8CtUxgn4sOB-26Ulgd3lYX1MhQ/0031qJTLJcE.jpg" 
alt="fall detection" width="360" height="180" border="10" /></a>



In this project, I tried to solve a problem using existing pose estimation deep learning models.
Current version works with CPU-based pose estimation model (Intel OpenVino pose estimation).

Some useful functions are:

>  convert_video_to_pics

generates pictures from video

>  check_anomalies

check whether fall happened in video
