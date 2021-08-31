syntax on             "语法高亮
filetype on           "文件类型检测

set encoding=utf-8    "Vim内部使用的字符编码
set hlsearch          "搜索结果高亮
set mouse=a           "支持使用鼠标
set nocp              "切断与vi的兼容
set showmode          "在底部显示，当前处于命令模式还是插入模式
set showcmd           "命令模式下，在底部显示当前键入的指令

set number            "显示行号
set relativenumber    "设置相对行号
set cursorline        "突出显示当前行
set textwidth         "设置行宽，一行显示多少字符
set wrap              "太长的会自动换行
set wrapmargin=2      "自动换行时，与右边缘空余的字符数
set ruler             "打开状态栏标志
set laststatus=2      "2显示状态栏，1只在多窗口时显示，0不显示

set autoindent        "下一行自动缩进为跟上一行一致
set smartindent       "智能缩进
set tabstop=3		    "tab键位为3
set expandtab         "自动将Tab转为空格
set softtabstop=3     "Tab转为3个空格
set shiftwidth=5      "一级缩进>>，取消一级缩进<<，取消全部缩进==时每一级的字符数

set ignorecase        "搜索时忽略大小写
set confirm           "没有保存或文件只读时弹出确认