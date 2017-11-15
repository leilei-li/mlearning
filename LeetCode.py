def step(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    else:
        return step(n - 1) + step(n - 2)


if __name__ == '__main__':
    n = input()
    print(step(int(n)))
