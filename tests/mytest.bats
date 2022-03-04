load harness

@test "mytest-1" {
  check 'if ( [5,15] < [-19, -7] ) then x := ( 49 ) * 3 + x else x := 1 * 2 * 2 + 3' '{x → 7}'
}

@test "mytest-2" {
  check 'x := ( 1 + 0 ) * -9' '{x → -9}'
}

@test "mytest-3" {
  check 'if ( 5 < 55 ) then x := [-991] else x := 119' '{x → [-991]}'
}

@test "mytest-4" {
  check 'while false do x := 1 ; array := [1, 2, -3]' '{array → [1, 2, -3]}'
}

@test "mytest-5" {
  check 'while false do x := 1 ; array := [1, -23, 101, -7]' '{array → [1, -23, 101, -7]}'
}