在终端中运行以下命令（）：

$DATA_DIR = "D:\Datasets"
$REPO_NAME = "AEC-Challenge"
$REPO_URL = "ssh://git@ssh.github.com:443/microsoft/AEC-Challenge.git"
$LFS_URL = "https://gh-proxy.com/https://github.com/microsoft/AEC-Challenge.git/info/lfs"
$TARGET_DIR = "datasets\synthetic"
$N = 270
$LIST_FILE = "selected_files.txt"

cd $DATA_DIR

Remove-Item -Recurse -Force ".\$REPO_NAME" -ErrorAction SilentlyContinue

$env:GIT_LFS_SKIP_SMUDGE="1"

git clone --filter=blob:none --sparse $REPO_URL $REPO_NAME

cd $REPO_NAME

git sparse-checkout set $TARGET_DIR

$selected = Get-ChildItem $TARGET_DIR -Directory | ForEach-Object {
    Get-ChildItem $_.FullName -File | Sort-Object Name | Select-Object -First $N
} | ForEach-Object {
    $_.FullName.Replace((Get-Location).Path + "\", "").Replace("\", "/")
}

$selected | Set-Content $LIST_FILE -Encoding ascii

(Get-Content $LIST_FILE).Count

Get-Content $LIST_FILE | git sparse-checkout set --no-cone --stdin

git config lfs.url $LFS_URL

git lfs env

Remove-Item Env:\GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue

git lfs install --force

Get-Content $LIST_FILE | ForEach-Object {
    Remove-Item -Force -ErrorAction SilentlyContinue $_
}

git checkout -f HEAD --pathspec-from-file=$LIST_FILE