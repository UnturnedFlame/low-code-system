from gradioAppDemo import Reactor
import multiprocessing
import time


def gradio_app_process():
    module_list = ['音频分离', '声纹识别']
    algorithm_dict = {'音频分离': 'sepformer', '声纹识别': 'transformer'}
    demo_app = Reactor()
    demo_app.start(module_list, algorithm_dict)
    demo_app.start(datafile='./algorithms/examples/mix.wav')
    demo_app.display(prevent_thread_lock=False)


if __name__ == '__main__':
    # p1 = multiprocessing.Process(group=None, target=gradio_app_process)
    # print('这是主进程')
    # time.sleep(1000)
    gradio_app_process()

