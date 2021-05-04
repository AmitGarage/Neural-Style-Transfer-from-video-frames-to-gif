# Neural-Style-Transfer-from-video-frames-to-gif
Extract Image frames from video to apply neural style and save converted images as gif

- Trained a CNN model ( VGG19 architecture before flattening ) on orignal image and style image as per paper on neural style transfer .
- Extracted image frames from video .
- Calulated content loss between original image and at every layer image pixel value.
- Calculated gram matrix over style image and find style loss between gram matrix and at every layer image pixel value.
- After every iteration style loss decreases as style is being imposed on original image.
- All converted images were saved as gif.

***Video used for extracting image frames and applying different style***

https://user-images.githubusercontent.com/17728616/117028882-bf4aba80-ad1b-11eb-9fee-8eb39c96edd9.mp4

***GIF formed from converted images***

![Video_to_gif](https://user-images.githubusercontent.com/17728616/117028131-05534e80-ad1b-11eb-956c-8d8afd6f65aa.gif)
