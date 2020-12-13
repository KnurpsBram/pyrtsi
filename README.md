# Real-Time Spectrogram Inversion (RTSI)

Python implementation of RTSI. The algorithm needs no iterations and no information about future spectrogram columns. It is ideal for true real-time ('streaming') audio conversion applications. The algorithm is inspired by RTISI-LA [(see paper)](https://ieeexplore.ieee.org/document/4244543). The algorithm inverts the spectrogram column-by-column, each time using the overlapping previous frames to make a good estimate of the phase of the current frame.  

Can be as simple as:
```
rtsi = RTSI(win_length=win_length, hop_length=hop_length)

spect       = rtsi.audio_to_spect(audio)
audio_recon = rtsi.spect_to_audio(spect)
```
See `simple_demo.ipynb` for more examples

## See also
http://ltfat.github.io/phaseret/doc/gabor/rtisila.html

## TO DO:
- [ ] enable multiple iterations
- [ ] enable lookahead
