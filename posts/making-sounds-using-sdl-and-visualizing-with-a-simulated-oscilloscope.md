<!--
.. title: Making sounds using SDL and visualizing them on a simulated oscilloscope.
.. slug: making-sounds-using-sdl-and-visualizing-them-on-a-simulated-oscilloscope
.. date: 2021-12-27 10:49:50 UTC+01:00
.. tags: 
.. category: 
.. link: 
.. description: 
.. type: text
.. pretty_url: False
-->

The most important part of [Vectron](https://studiostok.itch.io/vectron) is the real-time generation and visualization of audio. In the previous blog post we already built a simulated XY oscilloscope which we'll use to visualize the two stereo channels, and we'll use SDL for the platform-independent playback of the audio.
<!-- TEASER_END -->
# What is sound

To produce any sound on a digital computer, we first need to understand what sound is and how it is represented. In essence, sound is the collective vibration of particles. Our ears are made in such a way so as to sense these vibrations and convert them to electric signals that are then processed by our brains. A device that produces sound, like a speaker, does so by moving back and forth, transferring this motion to the air particles like a wave. It is this motion that our ears pick and can be naively visualized like in the image below.
<img src="/files/sound_waves.gif" style="display:block; padding:1em 0;"/>
The cyan particles is highlighted to show that, like in other types of waves, particles are not actually moving away from the sound source but oscillate around a fixed point. What you are actually perceive as motion, above, is the motion of the denser areas, also known as wave fronts. Note that you can also see sparse areas moving in the opposite direction. Our ear drum works in a similar manner; the air particles push the eardrum inwards when they are moving in the direction of our ears, while the eardrum, being elastic, oscillates back to its resting position when the velocity of the particles is small enough. Intuitively, you can probably see that the particle velocity is related to pressure; the larger the particle velocity when the particles reach the eardrum, the larger the pressure that is felt by the eardrum. Also, the larger the pressure, the louder the sound is perceived. Our ears can not only sense how loud a sound is, but it also senses its frequency which we perceive as the pitch of a sound -- the faster the particle oscillations, the higher the frequency, and thus the pitch.

# Sound in digital electronics

<img src="/files/oscillations.png" style="display:block; padding:1em 0;"/>
I mentioned that the eardrum converts the sound waves into electrical signal and that a speaker works in a similar way. Indeed, by supplying the speaker with alternating current of a certain frequency and amplitude, we can generate sound with a certain pitch and level, like in the picture above where two sinusoidal signals are depicted, with the cyan one having higher pitch, and smaller level, and the magenta one having a lower pitch but higher level. In analogue electronics the translation of sound into electrical signals is straightforward, but in digital electronics we only have 0s and 1s to represent the variation of sound over time, so we first have to 'digitize' sound. This is done by discretizing the signal into a fixed number of levels based on the number of bits we use. For example, an 8 bit sound can have 256 discrete levels. A computer then samples the current sound level with a fixed rate called the 'sampling rate', subsequently quantizes the level based on the number of bits, and this is then converted into an analogue signal to displace the cone of the speaker. The higher the level, the more the cone will be displaced, producing louder sounds. As you can imagine, if the sampling rate is too low, this will have an effect on how we perceive the sound. The most common audio sampling rate you'll see is 44.1kHz, or 44,100 samples per second. This is not an arbitrary number as humans can generally hear frequencies between 20Hz and 20kHz and the approximately double-rate is a requirement for perfect fidelity according to the [Nyquist theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem).

# Producing sound with SDL

Interfacing with an audio device in SDL is quite straightforward. An audio device can be 'opened' using the [SDL_OpenAudioDevice](https://wiki.libsdl.org/SDL_OpenAudioDevice) call:
```C++
SDL_AudioDeviceID SDL_OpenAudioDevice(
    const char *device,
    int iscapture,
    const SDL_AudioSpec *desired,
    SDL_AudioSpec *obtained,
    int allowed_changes
);
```
Setting `device = NULL` will give us back the most 'reasonable' default device. With the `SDL_AudioSpec* desired` argument we can specify the desired audio playback characteristics, like the frequency, format, number of channels, etc. Note that although computers are quite accurate, they cannot sample the sound consistently enough at the rates that are required, and thus in practice a buffer is introduced when interfacing between the software and the audio device. The size of the buffer can also be set in the desired `SDL_AudioSpec`, along with an audio callback function which is responsible for filling the buffer.

To make this all clear, let's build a small 'hello audio' program where we play a simple sinusoidal tone. Let's first write the main function:
```C++
#include <cstdint>
#include <SDL2/SDL.h>

int main(int argc, char* argv[])
{
    uint64_t samples_played = 0;

    SDL_AudioSpec audio_spec_want, audio_spec;
    SDL_memset(&audio_spec_want, 0, sizeof(audio_spec_want));

    audio_spec_want.freq     = 44100;
    audio_spec_want.format   = AUDIO_F32;
    audio_spec_want.channels = 2;
    audio_spec_want.samples  = 512;
    audio_spec_want.callback = audio_callback;
    audio_spec_want.userdata = (void*)&samples_played;

    SDL_AudioDeviceID SDL_OpenAudioDevice(
        NULL, 0,
        &audio_spec_want, &audio_spec,
        SDL_AUDIO_ALLOW_FORMAT_CHANGE
    );

    return 0;
}
```
which opens an audio device with a sampling rate of 44.1kHz and a buffer size of 512 samples. We use a 32-bit float representation for the audio, which allows us to represent audio as a floating point number between -1 and 1.
Next we implement the callback for filling the buffer:
```C++
void audio_callback(void* userdata, uint8_t* stream, int len)
{
    uint64_t* samples_played = (uint64_t*)userdata;
    float* fstream = (float*)(stream);
    static const float volume = 0.2;
    static const float frequency = 200.0;

    for(int sid = 0; sid < (len / 8); ++sid)
    {
        double time = (*samples_played + sid) / 44100.0;
        fstream[2 * sid + 0] = volume * sin(2.0 * M_PI * time * frequency); /* L */
        fstream[2 * sid + 1] = volume * sin(2.0 * M_PI * time * frequency); /* R */
    }

    *samples_played += (len / 8);
}
```
Here using `samples_played` we keep track of the progress in the audio playback so that we can sample at the appropriate times of our sinusoidal signal. Note that we divide `len / 8`, by the size of a single sample in bytes; because we chose for a 32-bit format with 2 channels, that's 4 bytes per channel.

<!--
TODO:
  * Push the program to github and link repo.
  * Ask to play with frequency and volume?
  * Connecting to oscilloscope, RingBuffer.
  * Conclusion.
-->

# Visualizing sound with an oscilloscope
