#! python3

def no_cover() :
  V = {"L1":0.0, "L2":0.0}
  cnt = 0

  for i in range(1000) :
    new_V = {}
    new_V['L1'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    new_V['L2'] = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])

    delta = max(abs(new_V['L1'] - V['L1']), abs(new_V['L2'] - V['L2']))
    V = new_V.copy()
    if delta < 0.0001 :
      break
    cnt += 1

  print("no_cover:")
  print(V)
  print(cnt)

def cover() :
  V = {"L1":0.0, "L2":0.0}
  cnt = 0

  for i in range(1000) :
    tmp_V = V.copy()
    V['L1'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    V['L2'] = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])

    delta = max(abs(tmp_V['L1'] - V['L1']), abs(tmp_V['L2'] - V['L2']))
    if delta < 0.0001 :
      break
    cnt += 1

  print("cover:")
  print(V)
  print(cnt)

no_cover()
cover()