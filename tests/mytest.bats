load harness

@test "mytest-1" {
  check 'x := false ? 10 : 22' '{x → 22}'
}

@test "mytest-2" {
  check 'x := true ? 5 : 11' '{x → 5}'
}

@test "mytest-3" {
  check 'x := 1 = 2 ? 99 : 55' '{x → 55}'
}

@test "mytest-4" {
  check 'x := 10 < 11 ? 1 * 1 : 20 / 2' '{x → 1}'
}

@test "mytest-5" {
  check 'x := 19 > 7 ? 55 / 5 : 5 * 11' '{x → 11}'
}
