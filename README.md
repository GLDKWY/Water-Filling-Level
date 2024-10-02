<h1 style="margin-left: 2em;width: auto;height: auto;">Understanding Dynamic Auditory Perception for Water Filling Level Estimation</h1>
<p style="margin-left: 4em;width: auto;height: auto;color: dodgerblue"> </p>
<p style="margin-left: 4em;width: auto;height: auto;"><b>Dataset:</b> Audio, Tactile and Visual perception for Filling Level Estimation（ATVfle）</p>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    The dataset consists of 1,140 audio-visual recordings with 12 human subjects manipulating 15 containers, split into 5 cups, 5 drinking glasses, and 5 food boxes. These containers are made of different materials, such as plastic, glass and paper. Each container can be empty or filled with water, rice or pasta at two different levels of fullness: 50% and 90% with respect to the capacity of the container. The combination of containers and fillings results in a total of 95 configurations acquired for three scenarios with an increasing level of difficulty, caused by occlusions or subject motions:</p>
<div align=center><img src="https://github.com/GLDKWY/Water-Filling-Level/blob/main/images/table1.png" style="margin-left: auto;margin-right:auto;display:block;width: 50%;height: 50%;"></div>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    Overview of data collection:</p>
<div align=center><img src="https://github.com/GLDKWY/Water-Filling-Level/blob/main/images/img1.png" style="margin-left: auto;margin-right:auto;display:block;width: 50%;height: 50%;"></div>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    Data collection demonstration video:    https://github.com/GLDKWY/Water-Filling-Level/assets/101457743/e6f6dbdb-a875-4cbc-be72-647543d7da47</p>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    Each scenario is recorded with two different backgrounds and under two different lighting conditions. The first background condition involves a plain tabletop with the subject wearing a texture-less t-shirt, while the second background condition involves the table covered with a graphics-printed tablecloth and the subject wearing a patterned shirt. The lighting conditions include ceiling room lights and controlled lights. The 95 configurations are executed by a different subject for each scenario and for each background/illumination condition.Each scenario is recorded with two different backgrounds and under two different lighting conditions. The first background condition involves a plain tabletop with the subject wearing a texture-less t-shirt, while the second background condition involves the table covered with a graphics-printed tablecloth and the subject wearing a patterned shirt. The lighting conditions include ceiling room lights and controlled lights. The 95 configurations are executed by a different subject for each scenario and for each background/illumination condition.</p>
<h1 style="margin-left: 2em;width: auto;height: auto;"></h1>
<p style="margin-left: 4em;width: auto;height: auto;">
    <b>Video aspect:</b></p>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    For the ease of calculating the percentage of water filling relative to the container capacity, we employed red dye in the water and positioned a Logitech C270 HD WEBCAM directly in front of the target container at a distance of 440 mm. Images were captured with a resolution of 1920 × 1080 at 24 fps.</p>
<h1 style="margin-left: 2em;width: auto;height: auto;"></h1>
<p style="margin-left: 4em;width: auto;height: auto;">
    <b>Audio aspect:</b></p>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    A BOYA BY-M1 microphone is attached to the left part of the jaw with a horizontal distance of 15 mm from the target container. Accordingly, variations in the audio signal throughout pouring trials can be captured.</p>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    Audio waveform and spectrum of water poured into glass in our dataset：</p>
<div align=center><img src="https://github.com/GLDKWY/Water-Filling-Level/blob/main/images/img2.png" style="margin-left: auto;margin-right:auto;display:block;width: 50%;height: 50%;"></div>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    Logarithmic Mel spectrogram converted from the original waveform：</p>
<div align=center><img src="https://github.com/GLDKWY/Water-Filling-Level/blob/main/images/img3.png" style="margin-left: auto;margin-right:auto;display:block;width: 50%;height: 50%;"></div>
<h1 style="margin-left: 2em;width: auto;height: auto;"></h1>
<p style="margin-left: 4em;width: auto;height: auto;">
    <b>Tactile aspect:</b></p>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    To assess the force variations applied to the grip jaw during the pouring process, we equipped a compact tactile sensor known as the L3 F-TOUCH which comprising a single chip, a micro-camera, and a trapezium-shaped rubber layer adorned with 40 black circular markers arranged in a 4 × 10 array.</p>
<div align=center><img src="https://github.com/GLDKWY/Water-Filling-Level/blob/main/images/img4.png" style="margin-left: auto;margin-right:auto;display:block;width: 50%;height: 50%;"></div>
<p style="text-indent:2em;word-wrap: break-word;word-break: break-word;margin-left: 4em;margin-right: 4em;">
    The display effect of the sensor is as follows：  https://github.com/GLDKWY/Water-Filling-Level/assets/101457743/9e09cf86-2682-4dbe-acda-a5b64201cd30</p>
