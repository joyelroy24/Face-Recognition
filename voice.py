# Import the required module for text
# to speech conversion
from gtts import gTTS,lang

# This module is imported so that we can
# play the converted audio
import os

# The text that you want to convert to audio
mytext = 'welcome to kochi'

# Language in which you want to convert
language = 'ml'
print(lang.tts_langs())

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)
print(type(myobj))
print(myobj)
# Saving the converted audio in a mp3 file named
# welcome
myobj.save("welcome.mp3")
# # Playing the converted file
os.system("welcome.mp3")
