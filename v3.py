import pyttsx3  
from translate import Translator
# initialize Text-to-speech engine  
engine = pyttsx3.init()  
voices = engine.getProperty('voices')
print(voices)
# engine.setProperty('voices',<pyttsx3.voice.Voice object at 0x000001A5799959E8>)
# convert this text to speech  
translator= Translator(to_lang="da")
translation = translator.translate("ambulance")
print(translation)

engine.say(translation)  
print(engine.getProperty("voices"))

# play the speech  
engine.runAndWait()  