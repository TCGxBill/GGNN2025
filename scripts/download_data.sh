#!/bin/bash

# ==============================================================================
# Smart PDB Downloader for Binding Site Prediction
#
# Tải dữ liệu một cách thông minh, ưu tiên các nguồn đáng tin cậy,
# có cơ chế dự phòng và hiển thị tiến trình.
# Author: Your Name
# Version: 4.1.0 (Đã sửa lỗi cấu trúc 'command not found')
# ==============================================================================

set -euo pipefail

# ==============================================================================
# SECTION 1: BIẾN TOÀN CỤC VÀ CÁC HÀM HỖ TRỢ
# ==============================================================================

DATA_DIR="data"
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# In tiêu đề cho một section
print_section() {
    echo -e "\n${BLUE}#${NC} ${YELLOW}$1${NC}"
    echo -e "${BLUE}# ====================================================================${NC}"
}

# Hàm dọn dẹp, sẽ được gọi tự động khi script kết thúc hoặc bị hủy
cleanup() {
    rm -rf /tmp/pdbbind_index_repo /tmp/pdbbind_ids.txt
}
trap cleanup EXIT

# Kiểm tra sự tồn tại của một lệnh
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Lỗi: Lệnh '$1' không được tìm thấy. Vui lòng cài đặt nó.${NC}"
        if [[ "$1" == "git-lfs" ]]; then
            echo -e "${YELLOW}Hướng dẫn cài đặt git-lfs:${NC}"
            echo "  - macOS:   brew install git-lfs"
            echo "  - Debian/Ubuntu: sudo apt-get install git-lfs"
            echo -e "Sau khi cài đặt, hãy chạy lệnh: ${GREEN}git lfs install${NC}"
        fi
        exit 1
    fi
}

# Hàm tải song song cốt lõi
run_parallel_download() {
    local pdb_list_str="$1"
    local output_dir="$2"
    local description="$3"
    
    read -r -a pdb_ids <<< "$pdb_list_str"
    local total=${#pdb_ids[@]}
    
    if [ "$total" -eq 0 ]; then
        echo -e "${YELLOW}Cảnh báo: Không có PDB ID nào để tải cho '${description}'.${NC}"
        return
    fi
    
    echo "Bắt đầu tải ${total} PDB files cho [${description}]..."
    mkdir -p "$output_dir"
    
    local cores
    cores=$(nproc 2>/dev/null || echo 8)
    
    download_one_pdb() {
        local pdb_id_upper
        pdb_id_upper=$(echo "$1" | tr '[:lower:]' '[:upper:]')
        local pdb_id_lower
        pdb_id_lower=$(echo "$1" | tr '[:upper:]' '[:lower:]')
        
        local dest_path="$2/${pdb_id_upper}.pdb"
        if [[ "$3" == "refined" ]]; then
            mkdir -p "$2/$1"
            dest_path="$2/$1/${1}_protein.pdb"
        fi
        
        if [ ! -f "$dest_path" ]; then
            # Try RCSB first (max time 10s)
            if ! curl -sfL --max-time 10 "https://files.rcsb.org/download/${pdb_id_upper}.pdb" -o "${dest_path}"; then
                # Fallback to EBI PDBe
                # echo "  RCSB failed for $1, trying EBI..."
                if ! curl -sfL --max-time 10 "https://www.ebi.ac.uk/pdbe/entry-files/pdb${pdb_id_lower}.ent" -o "${dest_path}"; then
                    echo "  Failed to download $1 from both RCSB and EBI."
                    rm -f "${dest_path}" # Cleanup empty file if created
                    return 1
                fi
            fi
        fi
    }
    export -f download_one_pdb

    printf "%s\n" "${pdb_ids[@]}" | xargs -n 1 -P "$cores" -I {} bash -c "download_one_pdb '{}' '$output_dir' '$description' && echo -n ."
    
    echo -e "\n${GREEN}✓ Hoàn tất tải [${description}]. Đã kiểm tra/tải ${total} files.${NC}"
}

# ==============================================================================
# SECTION 2: CÁC HÀM TẢI DATASET CỤ THỂ
# ==============================================================================

download_demo() {
    print_section "Demo Dataset"
    local pdbs="1ATP 1HVR 2BRD 3CPA 4TIM 1A28 2CPP 3PTB 1PPH 2PCB"
    run_parallel_download "$pdbs" "${DATA_DIR}/raw/demo" "demo"
}

download_pdbbind() {
    print_section "PDBbind Refined Set"
    local max_to_download=${1:-1000}
    local output_dir="${DATA_DIR}/raw/pdbbind/refined-set"
    local curated_list_file="${DATA_DIR}/pdbbind_curated_list.txt"

    echo "Cố gắng lấy danh sách PDBbind 2020 Refined Set chính thức từ GitHub..."
    local repo_dir="/tmp/pdbbind_index_repo"
    rm -rf "$repo_dir"
    
    if git clone --depth 1 "https://github.com/gregory-kyro/PDBbind_Protein_Only.git" "$repo_dir" &>/dev/null; then
        echo -e "${GREEN}✓ Lấy danh sách thành công.${NC}"
        cat "${repo_dir}/PDBbind_v2020_refined.txt" | grep -v '^#' | awk '{print $1}' > /tmp/pdbbind_ids.txt
    else
        echo -e "${YELLOW}Cảnh báo: Không thể lấy danh sách chính thức. Sử dụng danh sách dự phòng.${NC}"
        if [ ! -f "$curated_list_file" ]; then
            echo "Tạo file danh sách dự phòng tại '${curated_list_file}'..."
            cat > "$curated_list_file" << 'EOF'
1a1e 1a28 1a30 1a4k 1a9m 1abe 1adb 1add 1adl 1ai5 1ajp 1ajq 1ajx 1amo 1atl 1av1 1b38 1b39 1b40 1b58 1b8o 1bcu 1bjv 1bma 1bmk 1bnn 1br6 1bra 1bxo 1bxq 1c1b 1c1r 1c3x 1c5c 1c5z 1c70 1c83 1c87 1c88 1cbs 1cde 1cil 1cps 1ctr 1cvu 1d3h 1d3p 1d4l 1d7x 1dbj 1df8 1dmp 1dog 1dr1 1ds1 1dwb 1e1v 1e1x 1e66 1ebg 1ec0 1ec1 1ec2 1ec3 1ela 1elb 1elc 1eld 1ele 1ets 1ew8 1ez9 1f0r 1f0s 1f0t 1f0u 1f3d 1f3e 1f4e 1f4f 1f4g 1f57 1f5l 1f8b 1f8c 1f8d 1f8e 1fcx 1fcy 1fcz 1fgi 1fh7 1fh8 1fh9 1fjs 1fkg 1fki 1fl3 1flr 1fm6 1fm9 1fpc 1fq5 1fzk 1g2k 1g2l 1g32 1g35 1g3d 1g3e 1g45 1g46 1g4o 1g52 1g53 1g54 1g55 1g7f 1g7g 1g7q 1g7v 1g9v 1gaf 1ghy 1ghz 1gi0 1gi1 1gi2 1gi3 1gi4 1gi5 1gi6 1gi7 1gi8 1gj5 1gja 1gjc 1gkc 1gm8 1gpk 1gpl 1gsp 1h1l 1h22 1h23 1h46 1hbv 1hdc 1hf8 1hfc 1hih 1hii 1hk4 1hl7 1hms 1hnn 1hps 1hq2 1hsh 1hsl 1hvi 1hvj 1hvk 1hvl 1hvr 1hvs 1hwr 1hxb 1hxw 1hyt 1ia1 1ida 1ifb 1igj 1iiq 1ik4 1ivb 1ivc 1ivd 1ive 1ivf 1j16 1j17 1j4h 1jao 1jd0 1jdp 1jje 1jla 1jyq 1k1i 1k1j 1k1l 1k1m 1k1n 1k21 1k22 1k27 1k3u 1k4g 1ke5 1kpf 1kv1 1kv2 1kv5 1kzk 1l2s 1l7f 1lcb 1lcp 1lgt 1lhu 1li3 1li4 1lke 1lna 1lnm 1lpg 1lpm 1lpz 1lrh 1lst 1m0n 1m0o 1m0q 1m13 1m15 1m17 1m2q 1m2r 1m2z 1m48 1me8 1meh 1meu 1mfi 1mmv 1mnc 1mq6 1mrg 1mrk 1ms6 1msm 1msn 1mtp 1mts 1mu6 1mu8 1mui 1n1m 1n2j 1n2v 1n46 1n4h 1n5r 1nc1 1nc3 1nco 1ndw 1ndz 1nfy 1nh0 1nhz 1nj6 1nj8 1njb 1njs 1nl9 1nli 1nlj 1nny 1no6 1nvq 1nvr 1nvs 1nvt 1nvu 1nvv 1nvw 1nvx 1nvy 1o0h 1o2g 1o2h 1o2j 1o2z 1o30 1o3f 1o3i 1o3p 1o5b 1o86 1oai 1oda 1odj 1oew 1of1 1of6 1og1 1ogx 1oh0 1ohr 1oi9 1oir 1okl 1ol1 1ol2 1ol5 1oq5 1osg 1owe 1owh 1oxq 1oyt 1p1n 1p1o 1p1q 1p2y 1p62 1pme 1pmn 1pmv 1pr5 1ps3 1pso 1pw1 1pw2 1pxn 1pxo 1pxp 1q1g 1q41 1q4g 1q63 1q65 1q8t 1q8u 1qb1 1qbn 1qbq 1qbr 1qbs 1qbt 1qbu 1qf1 1qf2 1qi0 1qji 1qkt 1r0p 1r1h 1r5y 1r9l 1r9o 1re8 1rne 1ro6 1rpj 1s19 1s38 1s3v 1s63 1sgu 1sj0 1sl3 1slg 1sqn 1sqo 1sr7 1srj 1syi 1t40 1t46 1t4v 1t9b 1thl 1tng 1tni 1tnj 1tnk 1tnl 1tom 1tow 1tpp 1tt1 1tz8 1u1b 1u1c 1u33 1u4d 1uml 1unl 1uou 1ups 1utp 1uvs 1uto 1v0p 1v2n 1v2o 1v2p 1v2q 1v2r 1v2s 1v48 1v4s 1vcj 1vso 1vwn 1w1p 1w2g 1w3k 1w3l 1w4o 1xap 1xd0 1xgi 1xgj 1xid 1xie 1xif 1xih 1xk9 1xka 1xm6 1xoq 1xoz 1xr8 1y6r 1yc1 1yc4 1ydr 1yds 1ydt 1yei 1yej 1yqy 1yt9 1yv3 1yvm 1yvf 1ywn 1z95 1z9g 1zea 1zgi 1zoh 1zs0 2aoc 2aot 2aov 2ayr 2azr 2bm2 2boj 2bok 2br1 2brb 2brc 2bsm 2bvd 2bz6 2cbj 2cbz 2cbu 2cet 2cgr 2chz 2cht 2cji 2ctc 2d0k 2d1o 2d3u 2d3z 2er0 2er6 2er7 2er9 2erk 2f80 2f94 2fai 2fgi 2fgj 2fle 2fli 2fvd 2g00 2g70 2gss 2h15 2hb1 2hwi 2i0a 2i0d 2i78 2i80 2iik 2iwx 2j34 2j62 2j78 2j7h 2jdm 2jds 2nn7 2nta 2nw3 2o0u 2o4j 2o4p 2oag 2obj 2obp 2oiq 2ol1 2ovv 2p15 2p4y 2p7a 2pc8 2pog 2por 2pou 2pq9 2pqc 2pvl 2pvm 2pvu 2pxa 2pym 2q5k 2q64 2q7q 2qbp 2qbq 2qbr 2qe4 2qft 2qg0 2qg2 2qi5 2qnq 2r0u 2r23 2r9w 2rgp 2uwd 2uwl 2v00 2v3u 2v57 2v58 2v59 2v7a 2vj9 2vkm 2vot 2vvn 2vw5 2vwc 2w4x 2w66 2w6c 2wbg 2wca 2wer 2wgj 2wgx 2wij 2wka 2wn9 2wnc 2wos 2wtv 2wvt 2x00 2x8z 2x97 2xb8 2xbv 2xhm 2xii 2xj7 2xjg 2xmy 2xnb 2xys 2y5h 2yel 2yfe 2yge 2ygf 2yki 2ykj 2ykm 2ymd 2yme 2ypi 2ypj 2ypk 2ypl 2ypm 2ypn 2ypo 2ypp 2ypq 2ypr 2yps 2ypt 2ypu 2ypv 2ypw 2yq5 2yq7 2yqz 2zcq 2zcr 2zcs 2zct 2zdm 2zgx 2ziq 2zjw 2zxd 2zy1 3acw 3aid 3arq 3aru 3arv 3arw 3ary 3arz 3b1m 3b27 3b5r 3b65 3b68 3bgq 3bkk 3bpc 3bqc 3bv9 3bwj 3bxf 3bxh 3c2f 3c2u 3cft 3cj4 3coy 3cp9 3cpr 3cps 3cyx 3d0e 3d4z 3d6q 3d6t 3dd0 3dds 3dne 3drf 3dxg 3dyq 3e5a 3e6y 3e92 3e93 3ebp 3ebl 3ehy 3ekr 3el1 3el8 3elu 3eqc 3eqr 3ert 3f17 3f3a 3f3c 3f3d 3f3e 3f80 3fcq 3fk1 3fl5 3fur 3fv1 3fv2 3fx7 3g0w 3g2n 3g2z 3g31 3gbb 3gcs 3ggs 3gnw 3gy2 3gy4 3h30 3huc 3imc 3ivg 3jvr 3jvs 3jya 3kgp 3kwa 3l3n 3l4u 3l4w 3l7b 3lka 3n76 3n86 3n9s 3nq3 3nw9 3nxq 3oe4 3oe5 3ozt 3owj 3ozs 3ozt 3p5o 3p8n 3pe2 3prs 3pww 3pyy 3pyz 3qqs 3r17 3r88 3rfa 3rlr 3rr4 3rsx 3rv4 3ryj 3syr 3t08 3t09 3t0b 3t2q 3t64 3tsk 3u5j 3u8k 3u8n 3u9q 3udh 3ui7 3ueu 3uex 3uev 3uew 3uey 3uez 3uo4 3up2 3uri 3utu 3uzj 3vd4 3vdb 3vhc 3vri 3vw2 3wtj 3zso 3zsx 3zt2 4agn 4agp 4agq 4bkt 4crc 4de1 4de2 4de3 4djr 4djv 4dli 4ea2 4eor 4eos 4erf 4gfm 4gid 4gqq 4ivb 4ivc 4ivd 4jia 4lzs 4mme 4owm 4tmn 4ty7 4w9h 4w9i 4w9l 4wiv 4xiw 4xli 4xt2 4xuf 4yc7 4yfe 4zzi 5ab1 5c28 5std 5tmn 6abp 6cpa 6cts 6rsa 6std 6tmn 6upj 7cpa 7std 7tim 8abp 8cpa 8std 8tim 9abp 9std
EOF
        fi
        cp "$curated_list_file" /tmp/pdbbind_ids.txt
    fi
    
    local pdb_list_str=$(head -n "$max_to_download" /tmp/pdbbind_ids.txt | tr '\n' ' ')
    run_parallel_download "$pdb_list_str" "$output_dir" "refined"
}

download_coach420() {
    print_section "COACH420 Dataset"
    local pdbs="1a28 1a4k 1abf 1adb 1adl 1aec 1af6 1agw 1ai5 1aj6 1ajp 1ajq 1ajx 1akw 1amo 1aoe 1atl 1aw1 1b3l 1b38 1b58 1b8o 1bcu 1bjv 1bma 1bmk 1bnn 1br6 1bra 1bxo 1c1b 1c3x 1c5c 1c5z 1c70 1c83 1c87 1c88 1cbs 1cde 1cil 1cps 1ctr 1cvu 1d3h 1d3p 1d4l 1d7x 1dbj 1df8"
    run_parallel_download "$pdbs" "${DATA_DIR}/raw/coach420" "coach420"
}

download_holo4k() {
    print_section "Holo4k Benchmark Dataset"
    local max_to_download=${1:-1000}
    local output_dir="${DATA_DIR}/raw/holo4k"
    local list_url="https://raw.githubusercontent.com/rdk/p2rank-datasets/master/holo4k.ds"
    local list_file="${DATA_DIR}/holo4k_list.ds"
    
    echo "Downloading Holo4k list..."
    curl -sfL "$list_url" -o "$list_file"
    
    # Extract IDs: "holo4k/121p.pdb" -> "121p"
    local pdb_list_str=$(grep "holo4k/" "$list_file" | cut -d'/' -f2 | cut -d'.' -f1 | head -n "$max_to_download" | tr '\n' ' ')
    
    run_parallel_download "$pdb_list_str" "$output_dir" "holo4k"
}

download_joined() {
    print_section "Joined Training Dataset"
    local max_to_download=${1:-1000}
    local output_dir="${DATA_DIR}/raw/joined"
    local list_url="https://raw.githubusercontent.com/rdk/p2rank-datasets/master/joined.ds"
    local list_file="${DATA_DIR}/joined_list.ds"
    
    echo "Downloading Joined dataset list..."
    curl -sfL "$list_url" -o "$list_file"
    
    # Extract IDs: "joined/pdbbind/1a2k.pdb" -> "1a2k"
    # Use python for cross-platform shuffling (shuf is not always available on mac)
    local pdb_list_str=$(cat "$list_file" | awk -F'/' '{print $NF}' | cut -d'.' -f1 | sort | uniq | python3 -c "import sys, random; lines = sys.stdin.readlines(); random.shuffle(lines); print(''.join(lines), end='')" | head -n "$max_to_download" | tr '\n' ' ')
    
    run_parallel_download "$pdb_list_str" "$output_dir" "joined"
}

# ==============================================================================
# SECTION 3: LOGIC CHÍNH CỦA SCRIPT
# ==============================================================================

# Kiểm tra các lệnh cần thiết ngay từ đầu
check_command "git"
check_command "curl"
check_command "xargs"

# Tạo cấu trúc thư mục cơ bản
mkdir -p "${DATA_DIR}"/{raw/{pdbbind/refined-set,coach420,demo,holo4k,joined},processed,splits,cache}

# Lấy đối số từ dòng lệnh
dataset=${1:-demo}
max_proteins=${2:-1000}

# Chạy các hàm tương ứng dựa trên đối số
case $dataset in
    demo)
        download_demo
        ;;
    pdbbind)
        download_pdbbind "$max_proteins"
        ;;
    coach420)
        download_coach420
        ;;
    holo4k)
        download_holo4k "$max_proteins"
        ;;
    joined)
        download_joined "$max_proteins"
        ;;
    all)
        download_demo
        download_pdbbind "$max_proteins"
        download_coach420
        download_holo4k "$max_proteins"
        download_joined "$max_proteins"
        ;;
    *)
        echo -e "${RED}Lỗi: Dataset không hợp lệ: '$dataset'${NC}"
        echo "Usage: $0 [demo|pdbbind|coach420|holo4k|joined|all] [max_proteins]"
        exit 1
        ;;
esac

print_section "Hoàn tất"
echo "Tất cả các tác vụ tải đã xong."
echo "Bước tiếp theo: python scripts/preprocess_all.py --dataset $dataset"