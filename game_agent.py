import streamlit as st
import logging

logger = logging.getLogger(__name__)

def check_winner(board):
    lines = [
        # Rows 
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        # Columns
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        # Diagonals
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]]
    ]
    for line in lines:
        if line[0] != ' ' and all(cell == line[0] for cell in line):
            return line[0]  
    return None

def is_full(board):
    return all(cell != ' ' for row in board for cell in row)

def minimax(board, depth, is_max, max_depth):
    winner = check_winner(board)
    if winner == 'O':
        return 1
    elif winner == 'X':
        return -1
    elif is_full(board) or depth == max_depth:
        return 0

    if is_max:
        best = -float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    best = max(best, minimax(board, depth+1, False, max_depth))
                    board[i][j] = ' '
        return best
    else:
        best = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    best = min(best, minimax(board, depth+1, True, max_depth))
                    board[i][j] = ' '
        return best

def get_best_move(board, max_depth):
    best_val = -float('inf')
    best_move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False, max_depth)
                board[i][j] = ' '
                if move_val > best_val:
                    best_val = move_val
                    best_move = (i, j)
    return best_move

def make_move(i, j):
    if st.session_state.board[i][j] == ' ' and st.session_state.result == '':
        st.session_state.board[i][j] = 'X'
        logger.info(f"Player made move at position ({i}, {j})")
        
        winner = check_winner(st.session_state.board)
        if winner == 'X':
            st.session_state.result = "You win!"
            logger.info("Game ended - Player wins!")
            return
        elif is_full(st.session_state.board):
            st.session_state.result = "It's a tie!"
            logger.info("Game ended - It's a tie!")
            return

        current_emotion = st.session_state.get('emotion_state', {}).get('emotion', 'neutral').lower()
        
        max_depth = 5 if current_emotion in ['happy', 'neutral'] else 2
                
        ai_move = get_best_move(st.session_state.board, max_depth)
        
        if ai_move != (-1, -1):
            st.session_state.board[ai_move[0]][ai_move[1]] = 'O'
            logger.info(f"AI made move at position ({ai_move[0]}, {ai_move[1]})")
            
            winner = check_winner(st.session_state.board)
            if winner == 'O':
                st.session_state.result = "AI wins!"
                logger.info("Game ended - AI wins!")
            elif is_full(st.session_state.board):
                st.session_state.result = "It's a tie!"
                logger.info("Game ended - It's a tie!")

def play_game_ui():
    st.subheader("Play Tic-Tac-Toe Against Emotion-Aware AI")
    
    if "board" not in st.session_state:
        st.session_state.board = [[' ' for _ in range(3)] for _ in range(3)]
        st.session_state.turn = 'X'
        st.session_state.result = ''
        logger.info("New game initialized")
    

    def reset_game():
        st.session_state.board = [[' ' for _ in range(3)] for _ in range(3)]
        st.session_state.turn = 'X'
        st.session_state.result = ''
        logger.info("Game reset by player")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Reset Game", on_click=reset_game, use_container_width=True)

    st.markdown("### Game Board")
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            with cols[j]:
                cell_content = st.session_state.board[i][j]
                display_text = cell_content if cell_content != ' ' else " "
                
                if cell_content == ' ' and st.session_state.result == '':
                    st.button(
                        display_text, 
                        key=f"cell_{i}_{j}",
                        on_click=make_move,
                        args=(i, j),
                        use_container_width=True,
                    )
                else:
                    if cell_content == 'X':
                        st.markdown(f"<div style='text-align: center; font-size: 40px; color: blue; font-weight: bold; padding: 20px;'>{cell_content}</div>", unsafe_allow_html=True)
                    elif cell_content == 'O':
                        st.markdown(f"<div style='text-align: center; font-size: 40px; color: red; font-weight: bold; padding: 20px;'>{cell_content}</div>", unsafe_allow_html=True)
                    else:
                        st.button(
                            display_text, 
                            key=f"cell_{i}_{j}_disabled",
                            disabled=True,
                            use_container_width=True,
                        )

    if st.session_state.result:
        st.success(st.session_state.result)
            
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Your Symbol", "X")
    with col2:
        st.metric("AI Symbol", "O")
    