
def argmax(kv) :
  max_arg = None
  max_value = None
  for (k, v) in kv.items() :
    if max_arg is None or max_value is None or v > max_value:
      max_arg = k
      max_value = v
  return max_arg