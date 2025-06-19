from collections import deque

def is_valid(m_left, c_left):
    m_right = 3 - m_left
    c_right = 3 - c_left
    return (0 <= m_left <= 3 and 0 <= c_left <= 3 and
            0 <= m_right <= 3 and 0 <= c_right <= 3 and
            (m_left == 0 or m_left >= c_left) and
            (m_right == 0 or m_right >= c_right))

def get_successors(state):
    successors = []
    m, c, boat = state
    moves = [(2,0), (0,2), (1,0), (0,1), (1,1)]

    for m_move, c_move in moves:
        if boat == 'kanan':
            new_m = m + m_move
            new_c = c + c_move
            new_boat = 'kiri'
        else:
            new_m = m - m_move
            new_c = c - c_move
            new_boat = 'kanan'
        if is_valid(new_m, new_c):
            successors.append(((new_m, new_c, new_boat), (m_move, c_move)))
    return successors

def bfs():
    start = (0, 0, 'kanan')  
    goal = (3, 3, 'kiri')    

    queue = deque()
    queue.append((start, [start], [])) 
    visited = set()
    visited.add(start)

    while queue:
        current_state, path, moves = queue.popleft()

        if current_state == goal:
            return path, moves

        for next_state, move in get_successors(current_state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state], moves + [move]))

    return None, None


path, moves = bfs()

if path:
    print(f"Solusi ditemukan dalam {len(path) - 1} langkah:")
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        m_move, c_move = moves[i-1]

        aksi = []
        if m_move > 0:
            aksi.append(f"{m_move} misionaris")
        if c_move > 0:
            aksi.append(f"{c_move} kanibal")
        aksi_str = " dan ".join(aksi)

        arah = "menyeberang ke kiri" if prev[2] == 'kanan' else "kembali ke kanan"

        print(f"Langkah {i}: {prev} -> {curr}  # {aksi_str} {arah}")
else:
    print("Tidak ada solusi ditemukan.")
