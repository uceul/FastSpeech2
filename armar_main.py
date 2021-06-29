import subprocess
from armarx.slice_loader import load_armarx_slice
from armarx.ice_manager import register_object, using_topic, ice_communicator
load_armarx_slice('RobotAPI', 'speech/SpeechInterface.ice')
from armarx import TextListenerInterface


class MyTextListenerInterface(TextListenerInterface):
    def __init__(self):
        self.fastSpeechProcess = subprocess.Popen(['/home/mbehr/anaconda3/envs/fs_cpu/bin/python', '/home/mbehr/ws/FastSpeech2ARMAR/synthesize.py', '--restore_step', '900000', '--mode', 'stdin', '-p', 'config/LJSpeech/preprocess.yaml', '-m', 'config/LJSpeech/model.yaml', '-t', 'config/LJSpeech/train.yaml'], stdin=subprocess.PIPE)
    def reportText(self, text, current=None):
        print(text)
        self.fastSpeechProcess.stdin.write(text + '\n')
    def exitFestival(self):
        self.fastSpeechProcess.communicate()


def main():
    textToSpeechLister = MyTextListenerInterface()
    proxy = register_object(textToSpeechLister, 'NeuralTTS')
    using_topic(proxy, "TextToSpeech")
    iceCommunicator.waitForShutdown()
    textToSpeechLister.exitFestival()


if __name__ == '__main__':
    main()
