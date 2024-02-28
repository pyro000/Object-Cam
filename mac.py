import socket
import netifaces


def lettertonumber_andencript(letter):
    if not letter.isalpha():
        return letter
    for i, j in enumerate(range(97, 123)):
        if letter == chr(j):
            return f'-{i}-'


if __name__ == '__main__':

    var1 = ''
    local_ip = socket.gethostbyname(socket.gethostname())
    for nic in netifaces.interfaces():
        addrs = netifaces.ifaddresses(nic)
        try:
            if len(addrs[netifaces.AF_LINK][0]['addr']) > 0 and local_ip == addrs[netifaces.AF_INET][0]['addr']:
                var1 = addrs[netifaces.AF_LINK][0]['addr'].replace(':', '').zfill(12)
        except KeyError:
            pass

    var2 = [char for char in var1]
    var3 = ''
    for var in var2:
        var3 += lettertonumber_andencript(var)
    print(f'Returned: {var3}')
    input()
