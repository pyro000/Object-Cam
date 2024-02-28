class Key:
    def __init__(self, mac):
        self.letter = ''
        self.number = 0
        self.mac = mac
        self.dec = ''
        self.decript()
        self.key = self.createproductkey()

    def decript(self):
        f_l = False
        num = ''
        for word in [char for char in self.mac]:
            if f'{word}' == '-':
                if f_l:
                    f_l = False
                    self.dec += self.numbertoletter(num)
                    num = ''
                else:
                    f_l = True
            elif f_l:
                num += word
            else:
                self.dec += word

    def getkey(self):
        return self.key

    def lettertonumber(self, letter):
        self.letter = letter
        if not self.letter.isalpha():
            return self.letter
        for i, j in enumerate(range(97, 123)):
            if self.letter == chr(j):
                return i

    def numbertoletter(self, number):
        self.number = number
        if f'{self.number}'.isalpha():
            return self.number
        else:
            return chr(97 + int(self.number))

    def createproductkey(self):
        mac_c = []
        mac_c1 = []
        mac_c2 = []
        key = ''

        for i in self.dec:
            mac_c.append(i)
            mac_c1.append(self.lettertonumber(i))
            mac_c2.append(self.numbertoletter(i))

        mac_ci = mac_c[::-1]
        mac_ci1 = mac_c1[::-1]
        mac_ci2 = mac_c2[::-1]
        mac_f = [mac_c, mac_c1, mac_c2, mac_ci, mac_ci1, mac_ci2]

        key_ph = [[3, 2, 0], [1, 5, 1], [0, 7, 1], [5, 2, 1], [2, 8, 0], [3, 10, 1], [4, 4, 0], [3, 1, 0], [0, 3, 1],
                  [2, 7, 0], [3, 3, 0], [2, 6, 1], [4, 8, 1], [5, 7, 0], [0, 3, 0], [4, 9, 0], [5, 6, 0], [0, 7, 0],
                  [3, 9, 0], [1, 3, 0], [2, 7, 1], [5, 8, 1], [2, 1, 1], [1, 1, 0], [0, 11, 1], [0, 1, 1], [5, 7, 1],
                  [4, 8, 0], [2, 5, 0], [4, 0, 0], [5, 5, 1], [2, 3, 1], [5, 0, 0], [1, 10, 0], [2, 7, 0], [1, 3, 0],
                  [3, 7, 0], [3, 5, 1], [1, 5, 1], [3, 3, 1]]

        for i in key_ph:
            if i[2] == 1:
                if f'{mac_f[i[0]][i[1]]}' != 'i':
                    key += f'{mac_f[i[0]][i[1]]}'.upper()
            else:
                key += f'{mac_f[i[0]][i[1]]}'

        return key


if __name__ == '__main__':
    e_id = input('Ingrese e_id >')
    ok = Key(e_id)
    print(ok.getkey())
