# import threading as th

# def printEveryTwo():
#     print("every two seconds")

# def printEveryFive():
#     print("every five seconds")



# timeOne = (2.0, printEveryTwo)
# timeTwo = (5.0, printEveryFive)

# timer = th.Timer(5.0, printEveryFive)

# uid = 4

# chosen_time = timeOne if uid is None else timeTwo

# while True:
#     if not timer.is_alive():
#         timer = th.Timer(chosen_time[0], chosen_time[1])
#         timer.start()
    

