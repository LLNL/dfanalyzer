#include <algorithm>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <backward.hpp>
#include <bitset>
#include <codecvt>
#include <cpp-logger/logger.h>
#include <execinfo.h>
#include <experimental/filesystem>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>
#include <reader.h>
#include <reader-private.h>
#include <regex>
#include <set>
#include <signal.h>
#include <stdexcept>
#include <stdint-gcc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>

namespace fs = std::experimental::filesystem;

#define DFANALYZER_LOGGER cpplogger::Logger::Instance("DFANALYZER")
#define DFANALYZER_LOGDEBUG(format, ...) \
    DFANALYZER_LOGGER->log(cpplogger::LOG_DEBUG, format, __VA_ARGS__);
#define DFANALYZER_LOGINFO(format, ...) \
    DFANALYZER_LOGGER->log(cpplogger::LOG_INFO, format, __VA_ARGS__);
#define DFANALYZER_LOGWARN(format, ...) \
    DFANALYZER_LOGGER->log(cpplogger::LOG_WARN, format, __VA_ARGS__);
#define DFANALYZER_LOGERROR(format, ...) \
    DFANALYZER_LOGGER->log(cpplogger::LOG_ERROR, format, __VA_ARGS__);
#define DFANALYZER_LOGPRINT(format, ...) \
    DFANALYZER_LOGGER->log(cpplogger::LOG_PRINT, format, __VA_ARGS__);

std::string getexepath()
{
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    return std::string(result, (count > 0) ? count : 0);
}

std::string sh(std::string cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe)
        throw std::runtime_error("popen() failed!");
    while (!feof(pipe.get()))
    {
        if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
        {
            result += buffer.data();
        }
    }
    return result;
}

void signal_handler(int sig)
{
    switch (sig)
    {
    case SIGHUP:
    {
        DFANALYZER_LOGPRINT("hangup signal caught", 0);
        break;
    }
    case SIGTERM:
    {
        DFANALYZER_LOGPRINT("terminate signal caught", 0);
        MPI_Finalize();
        exit(0);
        break;
    }
    default:
    {
        backward::Printer p;
        backward::StackTrace st;
        st.load_here(32);
        p.print(st, std::cout);
        exit(1);
    }
    }
}

std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

class Directory
{
public:
    std::string directory;
    std::string hostname;
    std::string username;
    std::string app;
    int proc_id;
    double timestamp;
    Directory() : hostname(), username(), app(), proc_id(0), timestamp(0.0) {} /* default constructor */
    Directory(const Directory &other)
        : hostname(other.hostname),
          username(other.username),
          app(other.app),
          proc_id(other.proc_id),
          timestamp(other.timestamp) {} /* copy constructor*/
    Directory(Directory &&other)
        : hostname(other.hostname),
          username(other.username),
          app(other.app),
          proc_id(other.proc_id),
          timestamp(other.timestamp) {} /* move constructor*/

    Directory(std::string path)
    {
        this->directory = path;
        std::vector<std::string> split_strings = split(path, '-');
        if (split_strings.size() >= 4)
        {
            hostname = split_strings[0];
            username = split_strings[1];
            app = split_strings[2];
            for (int i = 3; i < split_strings.size() - 2; ++i)
            {
                app += "-" + split_strings[i];
            }
            proc_id = atoi(split_strings[split_strings.size() - 2].c_str());
            timestamp = atof(split_strings[split_strings.size() - 1].c_str());
        }
    }

    Directory &operator=(const Directory &other)
    {

        this->hostname = other.hostname;
        this->username = other.username;
        this->app = other.app;
        this->proc_id = other.proc_id;
        this->timestamp = other.timestamp;
        return *this;
    }

    bool operator==(const Directory &other) const
    {
        return this->timestamp == other.timestamp;
    }

    bool operator!=(const Directory &other) const
    {
        return this->timestamp != other.timestamp;
    }

    bool operator<(const Directory &other) const
    {
        return this->timestamp < other.timestamp;
    }

    bool operator>(const Directory &other) const
    {
        return this->timestamp > other.timestamp;
    }

    bool operator<=(const Directory &other) const
    {
        return this->timestamp <= other.timestamp;
    }

    bool operator>=(const Directory &other) const
    {
        return this->timestamp >= other.timestamp;
    }
};

struct ParquetWriter
{
    Directory directory;
    int rank;
    char *base_file;
    int64_t index;
    arrow::Int32Builder categoryBuilder, ioCategoryBuilder, rankBuilder, threadBuilder, levelBuilder, accessPatternBuilder;
    arrow::Int64Builder procBuilder, indexBuilder, sizeBuilder, fileIdBuilder, procIdBuilder, tmidBuilder;
    arrow::FloatBuilder tstartBuilder, tendBuilder, durationBuilder, bandwidthBuilder;
    arrow::StringBuilder func_idBuilder, appBuilder, hostnameBuilder, filenameBuilder, procnameBuilder;

    std::shared_ptr<arrow::Array> indexArray, categoryArray, ioCategoryArray, procArray, rankArray, threadArray, levelArray, tstartArray, tendArray,
        func_idArray, appArray, hostnameArray, sizeArray, durationArray, bandwidthArray, filenameArray, tmidArray, fileIdArray, procIdArray,
        accessPatternArray, procnameArray;

    std::shared_ptr<arrow::Schema> schema;
    const int64_t chunk_size = 1024;
    const int64_t NUM_ROWS = 1024 * 1024 * 1; // 1B
    int64_t row_group = 0;
    ParquetWriter(char *_path);
    void finish();
    int64_t max_tend;
    int64_t min_file_id, max_file_id;
    int64_t min_proc_id, max_proc_id;
    int64_t sum_transfer_size;
    float sum_bandwidth;
    int64_t record_count;
    std::map<int64_t, int64_t> file_offsets;
};
ParquetWriter::ParquetWriter(char *_path) : max_tend(0),
                                            sum_transfer_size(0), sum_bandwidth(0),
                                            record_count(0)
{

    min_file_id = std::numeric_limits<std::int64_t>::max();
    max_file_id = std::numeric_limits<std::int64_t>::min();
    min_proc_id = std::numeric_limits<std::int64_t>::max();
    max_proc_id = std::numeric_limits<std::int64_t>::min();
    row_group = 0;
    base_file = _path;
    schema = arrow::schema({arrow::field("index", arrow::int64()),
                            arrow::field("proc", arrow::int64()),
                            arrow::field("rank", arrow::int32()),
                            arrow::field("thread_id", arrow::int32()),
                            arrow::field("cat", arrow::int32()),
                            arrow::field("io_cat", arrow::int32()),
                            arrow::field("tstart", arrow::float32()),
                            arrow::field("tend", arrow::float32()),
                            arrow::field("func_id", arrow::utf8()),
                            arrow::field("level", arrow::int32()),
                            arrow::field("hostname", arrow::utf8()),
                            arrow::field("app", arrow::utf8()),
                            arrow::field("proc_name", arrow::utf8()),
                            arrow::field("file_name", arrow::utf8()),
                            arrow::field("size", arrow::int64()),
                            arrow::field("acc_pat", arrow::int32()),
                            arrow::field("bandwidth", arrow::float32()),
                            arrow::field("duration", arrow::float32()),
                            arrow::field("tmid", arrow::int64()),
                            arrow::field("file_id", arrow::int64()),
                            arrow::field("proc_id", arrow::int64())});
    index = 0;

    indexBuilder = arrow::Int64Builder();
    indexArray.reset();
    procIdBuilder = arrow::Int64Builder();
    procIdArray.reset();
    fileIdBuilder = arrow::Int64Builder();
    fileIdArray.reset();
    procBuilder = arrow::Int64Builder();
    procArray.reset();
    sizeBuilder = arrow::Int64Builder();
    sizeArray.reset();
    tmidBuilder = arrow::Int64Builder();
    tmidArray.reset();

    rankBuilder = arrow::Int32Builder();
    rankArray.reset();
    threadBuilder = arrow::Int32Builder();
    threadArray.reset();
    categoryBuilder = arrow::Int32Builder();
    categoryArray.reset();
    ioCategoryBuilder = arrow::Int32Builder();
    ioCategoryArray.reset();
    levelBuilder = arrow::Int32Builder();
    levelArray.reset();
    accessPatternBuilder = arrow::Int32Builder();
    accessPatternArray.reset();

    hostnameBuilder = arrow::StringBuilder();
    hostnameArray.reset();
    procnameBuilder = arrow::StringBuilder();
    procnameArray.reset();
    filenameBuilder = arrow::StringBuilder();
    filenameArray.reset();

    tstartBuilder = arrow::FloatBuilder();
    tstartArray.reset();
    tendBuilder = arrow::FloatBuilder();
    tendArray.reset();
    durationBuilder = arrow::FloatBuilder();
    durationArray.reset();
    bandwidthBuilder = arrow::FloatBuilder();
    bandwidthArray.reset();

    func_idBuilder = arrow::StringBuilder();
    func_idArray.reset();
    appBuilder = arrow::StringBuilder();
    appArray.reset();
}

void ParquetWriter::finish(void)
{
    procIdBuilder.Finish(&procIdArray);
    fileIdBuilder.Finish(&fileIdArray);
    indexBuilder.Finish(&indexArray);
    procBuilder.Finish(&procArray);
    sizeBuilder.Finish(&sizeArray);
    rankBuilder.Finish(&rankArray);
    threadBuilder.Finish(&threadArray);
    tstartBuilder.Finish(&tstartArray);
    tendBuilder.Finish(&tendArray);
    durationBuilder.Finish(&durationArray);
    bandwidthBuilder.Finish(&bandwidthArray);
    tmidBuilder.Finish(&tmidArray);
    func_idBuilder.Finish(&func_idArray);
    levelBuilder.Finish(&levelArray);
    hostnameBuilder.Finish(&hostnameArray);
    procnameBuilder.Finish(&procnameArray);
    filenameBuilder.Finish(&filenameArray);
    appBuilder.Finish(&appArray);
    categoryBuilder.Finish(&categoryArray);
    ioCategoryBuilder.Finish(&ioCategoryArray);
    accessPatternBuilder.Finish(&accessPatternArray);

    auto table = arrow::Table::Make(schema, {indexArray,
                                             procArray,
                                             rankArray,
                                             threadArray,
                                             categoryArray,
                                             ioCategoryArray,
                                             tstartArray,
                                             tendArray,
                                             func_idArray,
                                             levelArray,
                                             hostnameArray,
                                             appArray,
                                             procnameArray,
                                             filenameArray,
                                             sizeArray,
                                             accessPatternArray,
                                             bandwidthArray,
                                             durationArray,
                                             tmidArray,
                                             fileIdArray,
                                             procIdArray});

    char path[256];
    sprintf(path, "%s_%d.parquet", base_file, row_group);
    PARQUET_ASSIGN_OR_THROW(auto outfile, arrow::io::FileOutputStream::Open(path));
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024));
}

RecorderReader reader;

char *get_record_arg(const char *func_name, Record *record, int arg_index)
{
    if (sizeof(record->args) > arg_index)
    {
        return record->args[arg_index];
    }
    std::cout << "Unknown get_filename condition for func_name: " << func_name << std::endl;
    return "";
}

char *get_filename(Record *record)
{
    int cat = recorder_get_func_type(&reader, record);
    if (cat == RECORDER_FTRACE)
    {
        return "";
    }
    const char *func_name = recorder_get_func_name(&reader, record);
    const char *open_condition = strstr(func_name, "open");
    const char *opendir_condition = strstr(func_name, "opendir");
    const char *mpi_condition = strstr(func_name, "MPI");
    const char *close_condition = strstr(func_name, "close");
    const char *closedir_condition = strstr(func_name, "closedir");
    const char *fread_condition = strstr(func_name, "fread");
    const char *fwrite_condition = strstr(func_name, "fwrite");
    const char *read_condition = strstr(func_name, "read");
    const char *readdir_condition = strstr(func_name, "readdir");
    const char *readlink_condition = strstr(func_name, "readlink");
    const char *write_condition = strstr(func_name, "write");
    const char *xstat_condition = strstr(func_name, "__xstat");
    const char *fxstat_condition = strstr(func_name, "__fxstat");
    const char *lxstat_condition = strstr(func_name, "__lxstat");
    const char *lseek_condition = strstr(func_name, "lseek");
    const char *fseek_condition = strstr(func_name, "fseek");
    const char *ftruncate_condition = strstr(func_name, "ftruncate");
    const char *vfprintf_condition = strstr(func_name, "vfprintf");
    const char *remove_condition = strstr(func_name, "remove");
    const char *unlink_condition = strstr(func_name, "unlink");
    const char *access_condition = strstr(func_name, "access");
    const char *fileno_condition = strstr(func_name, "fileno");
    const char *ftell_condition = strstr(func_name, "ftell");
    const char *getcwd_condition = strstr(func_name, "getcwd");
    const char *mkdir_condition = strstr(func_name, "mkdir");
    const char *fcntl_condition = strstr(func_name, "fcntl");
    const char *rmdir_condition = strstr(func_name, "rmdir");
    const char *chmod_condition = strstr(func_name, "chmod");
    if (opendir_condition)
        return record->args[0];
    if (open_condition && !mpi_condition)
        return record->args[0];
    if (open_condition && mpi_condition)
        return get_record_arg(func_name, record, 1);
    if (close_condition && !closedir_condition)
        return record->args[0];
    if (read_condition && !fread_condition && !readdir_condition && !readlink_condition)
        return record->args[0];
    if (fread_condition)
        return get_record_arg(func_name, record, 3);
    if (write_condition && !fwrite_condition)
        return record->args[0];
    if (fwrite_condition)
        return get_record_arg(func_name, record, 3);
    if (xstat_condition)
        return get_record_arg(func_name, record, 1);
    if (fxstat_condition)
        return get_record_arg(func_name, record, 1);
    if (lxstat_condition)
        return get_record_arg(func_name, record, 1);
    if (lseek_condition)
        return record->args[0];
    if (fseek_condition)
        return record->args[0];
    if (ftruncate_condition)
        return record->args[0];
    if (vfprintf_condition)
        return record->args[0];
    if (remove_condition)
        return record->args[0];
    if (unlink_condition)
        return record->args[0];
    if (access_condition)
        return record->args[0];
    if (fileno_condition)
        return record->args[0];
    if (ftell_condition)
        return record->args[0];
    if (mkdir_condition)
        return record->args[0];
    if (fcntl_condition)
        return record->args[1];
    if (rmdir_condition)
        return record->args[1];
    if (chmod_condition)
        return record->args[1];
    if (!std::string(func_name).rfind("H5", 0) == 0 && !mpi_condition && !getcwd_condition)
    {
        // if (record->args) {
        //     if (sizeof(record->args) > 1 && record->args[0] && record->args[1]) {
        //         std::cout << "Unknown get_filename condition for func_name: " << func_name << " arg0: " << record->args[0] << " arg1: " << record->args[1]  << std::endl;
        //     } else if (sizeof(record->args) == 1 && record->args[0]) {
        //         std::cout << "Unknown get_filename condition for func_name: " << func_name << " arg0: " << record->args[0] << std::endl;
        //     } else {
        //         std::cout << "Unknown get_filename condition for func_name: " << func_name << std::endl;
        //     }
        // } else {
        //     std::cout << "Unknown get_filename condition for func_name: " << func_name << std::endl;
        // }
        // std::cout << "Unknown get_filename condition for func_name: " << func_name << std::endl;
    }
    return "UNKNOWN";
}

int64_t get_size(Record *record)
{
    int cat = recorder_get_func_type(&reader, record);
    if (cat == RECORDER_FTRACE)
    {
        return 0;
    }
    const char *func_name = recorder_get_func_name(&reader, record);
    const char *open_condition = strstr(func_name, "open");
    const char *mpi_condition = strstr(func_name, "MPI");
    const char *fread_condition = strstr(func_name, "fread");
    const char *close_condition = strstr(func_name, "close");
    const char *fwrite_condition = strstr(func_name, "fwrite");
    const char *read_condition = strstr(func_name, "read");
    const char *write_condition = strstr(func_name, "write");
    const char *readdir_condition = strstr(func_name, "readdir");
    const char *readlink_condition = strstr(func_name, "readlink");
    const char *lseek_condition = strstr(func_name, "lseek");
    const char *fseek_condition = strstr(func_name, "fseek");
    if (read_condition && !fread_condition && !readdir_condition && !readlink_condition)
        return atol(record->args[2]);
    if (fread_condition)
        return atol(record->args[2]);
    if (write_condition && !fwrite_condition)
        return atol(record->args[2]);
    if (fwrite_condition)
        return atol(record->args[2]);
    if (lseek_condition || fseek_condition)
        return atol(record->args[1]);
    // std::cout << "Unknown get_size condition for func_name: " << func_name << std::endl;
    return 0;
}

int64_t get_count(Record *record)
{
    int cat = recorder_get_func_type(&reader, record);
    if (cat == RECORDER_FTRACE)
    {
        return 0;
    }
    const char *func_name = recorder_get_func_name(&reader, record);
    const char *fread_condition = strstr(func_name, "fread");
    const char *fwrite_condition = strstr(func_name, "fwrite");
    if (fread_condition)
        return atol(record->args[1]);
    if (fwrite_condition)
        return atol(record->args[1]);
    // std::cout << "Unknown get_count condition for func_name: " << func_name << " arg0: " << record->args[0] << " arg1: " << record->args[1]  << std::endl;
    return 1;
}

void trim_utf8(std::string &hairy)
{
    std::vector<bool> results;
    std::string smooth;
    size_t len = hairy.size();
    results.reserve(len);
    smooth.reserve(len);
    const unsigned char *bytes = (const unsigned char *)hairy.c_str();

    auto read_utf8 = [](const unsigned char *bytes, size_t len, size_t *pos) -> unsigned
    {
        int code_unit1 = 0;
        int code_unit2, code_unit3, code_unit4;

        if (*pos >= len)
            goto ERROR1;
        code_unit1 = bytes[(*pos)++];

        if (code_unit1 < 0x80)
            return code_unit1;
        else if (code_unit1 < 0xC2)
            goto ERROR1; // continuation or overlong 2-byte sequence
        else if (code_unit1 < 0xE0)
        {
            if (*pos >= len)
                goto ERROR1;
            code_unit2 = bytes[(*pos)++]; // 2-byte sequence
            if ((code_unit2 & 0xC0) != 0x80)
                goto ERROR2;
            return (code_unit1 << 6) + code_unit2 - 0x3080;
        }
        else if (code_unit1 < 0xF0)
        {
            if (*pos >= len)
                goto ERROR1;
            code_unit2 = bytes[(*pos)++]; // 3-byte sequence
            if ((code_unit2 & 0xC0) != 0x80)
                goto ERROR2;
            if (code_unit1 == 0xE0 && code_unit2 < 0xA0)
                goto ERROR2; // overlong
            if (*pos >= len)
                goto ERROR2;
            code_unit3 = bytes[(*pos)++];
            if ((code_unit3 & 0xC0) != 0x80)
                goto ERROR3;
            return (code_unit1 << 12) + (code_unit2 << 6) + code_unit3 - 0xE2080;
        }
        else if (code_unit1 < 0xF5)
        {
            if (*pos >= len)
                goto ERROR1;
            code_unit2 = bytes[(*pos)++]; // 4-byte sequence
            if ((code_unit2 & 0xC0) != 0x80)
                goto ERROR2;
            if (code_unit1 == 0xF0 && code_unit2 < 0x90)
                goto ERROR2; // overlong
            if (code_unit1 == 0xF4 && code_unit2 >= 0x90)
                goto ERROR2; // > U+10FFFF
            if (*pos >= len)
                goto ERROR2;
            code_unit3 = bytes[(*pos)++];
            if ((code_unit3 & 0xC0) != 0x80)
                goto ERROR3;
            if (*pos >= len)
                goto ERROR3;
            code_unit4 = bytes[(*pos)++];
            if ((code_unit4 & 0xC0) != 0x80)
                goto ERROR4;
            return (code_unit1 << 18) + (code_unit2 << 12) + (code_unit3 << 6) + code_unit4 - 0x3C82080;
        }
        else
            goto ERROR1; // > U+10FFFF

    ERROR4:
        (*pos)--;
    ERROR3:
        (*pos)--;
    ERROR2:
        (*pos)--;
    ERROR1:
        return code_unit1 + 0xDC00;
    };

    unsigned c;
    size_t pos = 0;
    size_t pos_before;
    size_t inc = 0;
    bool valid;

    for (;;)
    {
        pos_before = pos;
        c = read_utf8(bytes, len, &pos);
        inc = pos - pos_before;
        if (!inc)
            break; // End of string reached.

        valid = false;

        if ((c <= 0x00007F) || (c >= 0x000080 && c <= 0x0007FF) || (c >= 0x000800 && c <= 0x000FFF) || (c >= 0x001000 && c <= 0x00CFFF) || (c >= 0x00D000 && c <= 0x00D7FF) || (c >= 0x00E000 && c <= 0x00FFFF) || (c >= 0x010000 && c <= 0x03FFFF) || (c >= 0x040000 && c <= 0x0FFFFF) || (c >= 0x100000 && c <= 0x10FFFF))
            valid = true;

        if (c >= 0xDC00 && c <= 0xDCFF)
        {
            valid = false;
        }

        do
            results.push_back(valid);
        while (--inc);
    }

    size_t sz = results.size();
    for (size_t i = 0; i < sz; ++i)
    {
        if (results[i])
            smooth.append(1, hairy.at(i));
    }

    hairy.swap(smooth);
}

#define OTHER_FUNC 0
#define READ_FUNC 1
#define WRITE_FUNC 2
#define METADATA_FUNC 3

int get_io_category(Record *record)
{
    const char *func_id = recorder_get_func_name(&reader, record);
    int cat = recorder_get_func_type(&reader, record);
    if (cat == 0 || cat == 1 || cat == 3)
    {
        // IO Category
        bool closedir_condition = std::string(func_id).find("closedir") != std::string::npos;
        bool getcwd_condition = std::string(func_id).find("getcwd") != std::string::npos;
        bool read_condition = std::string(func_id).find("read") != std::string::npos;
        bool readdir_condition = std::string(func_id).find("readdir") != std::string::npos;
        bool readlink_condition = std::string(func_id).find("readlink") != std::string::npos;
        bool umask_condition = std::string(func_id).find("umask") != std::string::npos;
        bool write_condition = std::string(func_id).find("write") != std::string::npos;
        if (closedir_condition || getcwd_condition || readdir_condition || readlink_condition || umask_condition)
            return OTHER_FUNC;
        else if (read_condition)
            return READ_FUNC;
        else if (write_condition)
            return WRITE_FUNC;
        else
            return METADATA_FUNC;
    }
    else
    {
        return OTHER_FUNC;
    }
}
void handle_eptr(std::exception_ptr eptr) // passing by value is ok
{
    try
    {
        if (eptr)
        {
            std::rethrow_exception(eptr);
        }
    }
    catch (const std::exception &e)
    {
        DFANALYZER_LOGERROR("Caught exception:  %s", e.what());
    }
}

void handle_one_record(Record *record, void *arg)
{
    try
    {
        ParquetWriter *writer = (ParquetWriter *)arg;
        int cat = recorder_get_func_type(&reader, record);
        const char *func_id = recorder_get_func_name(&reader, record);
        double duration = record->tend - record->tstart;
        uint64_t tmid = (record->tend + record->tstart) / 2.0 / reader.metadata.time_resolution;
        uint64_t tend = record->tend / reader.metadata.time_resolution;
        if (writer->max_tend < tend)
            writer->max_tend = tend;
        std::string file = "UNKNOWN";
        int64_t size = 0;
        int64_t count = 1;
        double bandwidth = 0.0;
        int64_t process_hash = 0;
        int io_cat = OTHER_FUNC;
        int access_pattern = 0;
        int64_t file_hash = 0;
        std::string proc_name = "UNKNOWN";
        if (cat == 0)
        {
            file = std::string(get_filename(record));
            {
                fs::path file_path;
                if (!file.empty() or file != "UNKNOWN")
                {
                    file_path = fs::path(file);
                    if (!file_path.has_parent_path())
                    {
                        if (const char *env_working_dir = std::getenv("R2P_WORKING_DIR"))
                        {
                            fs::path working_dir = fs::path(env_working_dir);
                            fs::path full_path = working_dir / file_path;
                            file = full_path.string();
                        }
                    }
                }
                std::string::difference_type n = std::count(file.begin(), file.end(), '/');

                if (!file.empty() && n > 2)
                {
                    fs::path filepath = fs::path(file);
                    auto filename = filepath.filename().string();
                    auto dir_0 = filepath.parent_path();
                    auto dir_0_name = dir_0.filename().string();
                    auto dir_1 = dir_0.parent_path();
                    auto dir_1_name = dir_1.filename().string();
                    auto dir_2 = dir_1.parent_path().string();
                    int64_t filename_hash =
                        std::hash<std::string>()(filename) % std::numeric_limits<std::uint32_t>::max();
                    // std::bitset<64> x(filename_hash);
                    // std::cout << x << '\n';
                    int64_t dir_0_hash =
                        (std::hash<std::string>()(dir_0_name) % std::numeric_limits<std::uint16_t>::max()) << 32;
                    // x = std::bitset<64> (dir_0_hash);
                    // std::cout << x << '\n';
                    int64_t dir_1_hash =
                        (std::hash<std::string>()(dir_1_name) % std::numeric_limits<std::uint8_t>::max()) << 48;
                    // x= std::bitset<64> (dir_1_hash);
                    // std::cout << x << '\n';
                    int64_t dir_2_hash =
                        (std::hash<std::string>()(dir_2) % std::numeric_limits<std::int8_t>::max()) << 56;
                    // x= std::bitset<64> (dir_2_hash);
                    // std::cout << x << '\n';
                    file_hash = file_hash | filename_hash | dir_0_hash | dir_1_hash | dir_2_hash;
                    // x= std::bitset<64> (hash);
                    // std::cout << x << '\n';
                }
                else if (file.find("/dev/") == 0)
                {
                    file_hash = std::hash<std::string>()(file);
                }
                else
                {
                    file_hash = std::hash<std::string>()(file);
                }
                // trim_utf8(file);
                if (writer->min_file_id > file_hash)
                    writer->min_file_id = file_hash;
                if (writer->max_file_id < file_hash)
                    writer->max_file_id = file_hash;
            }
            io_cat = get_io_category(record);
            size = get_size(record);
            count = get_count(record);
            size = size * count;
            bandwidth = duration > 0 ? (size * 1.0 / duration / 1024.0 / 1024.0) : 0.0;

            int64_t thread_hash = (std::hash<int>()(record->tid) % std::numeric_limits<std::uint16_t>::max());
            int process_id_used = writer->directory.proc_id;
            if (process_id_used == 1)
                process_id_used = writer->rank;
            proc_name = writer->directory.app + "#" +
                        writer->directory.hostname + "#" +
                        std::to_string(process_id_used) + "#" +
                        std::to_string(record->tid);

            int64_t process_id_hash = (std::hash<int>()(process_id_used) % std::numeric_limits<std::uint16_t>::max()) << 16;
            int64_t hostname_hash = (std::hash<std::string>()(writer->directory.hostname) % std::numeric_limits<std::int16_t>::max()) << 32;
            int64_t app_hash = (std::hash<std::string>()(writer->directory.app) % std::numeric_limits<std::uint16_t>::max()) << 48;
            process_hash = app_hash | hostname_hash | process_id_hash | thread_hash;

            if (writer->min_proc_id > process_hash)
                writer->min_proc_id = process_hash;
            if (writer->max_proc_id < process_hash)
                writer->max_proc_id = process_hash;

            if (writer->file_offsets.find(file_hash) == writer->file_offsets.end())
            {
                writer->file_offsets[file_hash] = size;
            }
            else
            {
                const char *open_condition = strstr(func_id, "open");
                const char *opendir_condition = strstr(func_id, "opendir");
                const char *close_condition = strstr(func_id, "close");

                const char *fopen_condition = strstr(func_id, "fopen");
                const char *fclose_condition = strstr(func_id, "fclose");
                const char *ftruncate_condition = strstr(func_id, "ftruncate");

                const char *lseek_condition = strstr(func_id, "lseek");
                const char *fseek_condition = strstr(func_id, "fseek");

                int64_t current_offset = writer->file_offsets[file_hash];
                if (std::string(func_id).rfind("H5", 0) == 0 || std::string(func_id).rfind("MPI", 0) == 0)
                {
                    // ignore if it is a H5 or MPI call
                }
                else if (lseek_condition || fseek_condition)
                {
                    if (current_offset != size)
                    {
                        access_pattern = 1; // random
                    }
                    writer->file_offsets[file_hash] = size; // seek position
                }
                else
                {
                    if (fopen_condition)
                    {
                        std::string open_mode = std::string(record->args[1]);
                        if (open_mode.rfind("r", 0) == 0 || open_mode.rfind("w", 0) == 0)
                        { // read or write
                            current_offset = 0;
                        }
                        else
                        {
                            if (sizeof(record->args) > 1)
                            {
                                std::cout << "Unknown fopen_condition arg0: " << record->args[0] << " arg1: " << record->args[1] << std::endl;
                            }
                            else
                            {
                                std::cout << "Unknown fopen_condition arg0: " << record->args[0] << std::endl;
                            }
                        }
                    }
                    else if (open_condition && !opendir_condition)
                    {
                        int open_flag = atol(record->args[1]); // get open flags
                        // The file creation flags are O_CLOEXEC, O_CREAT, O_DIRECTORY, O_EXCL, O_NOCTTY, O_NOFOLLOW, O_TMPFILE, and O_TRUNC
                        if (open_flag & O_CLOEXEC || open_flag & O_CREAT || open_flag & O_DIRECTORY || open_flag & O_EXCL ||
                            open_flag & O_NOCTTY || open_flag & O_NOFOLLOW || open_flag & O_TMPFILE || open_flag & O_TRUNC)
                        {
                            // std::cout << "offset zeroed because: " << open_flag << std::endl;
                            current_offset = 0;
                        }
                        else
                        {
                            // if (sizeof(record->args) > 1) {
                            //     std::cout << "Unknown " << func_id << " arg0: " << record->args[0] << " arg1: " << record->args[1] << std::endl;
                            // } else {
                            //     std::cout << "Unknown " << func_id << " arg0: " << record->args[0] << std::endl;
                            // }
                        }
                    }
                    else if (fclose_condition || close_condition || ftruncate_condition)
                    {
                        current_offset = 0;
                    }
                    writer->file_offsets[file_hash] = current_offset + size; // change offset
                }
            }

            writer->sum_transfer_size += size;
            writer->sum_bandwidth += bandwidth;
        }
        writer->record_count++;
        writer->durationBuilder.Append(duration);
        writer->filenameBuilder.Append(file);
        writer->sizeBuilder.Append(size);
        writer->bandwidthBuilder.Append(bandwidth);
        writer->tmidBuilder.Append(tmid);
        writer->procBuilder.Append(writer->directory.proc_id);
        writer->rankBuilder.Append(writer->rank);
        writer->threadBuilder.Append(record->tid);
        writer->tstartBuilder.Append(record->tstart);
        writer->tendBuilder.Append(record->tend);
        if (cat == RECORDER_FTRACE)
        {
            writer->func_idBuilder.Append(record->args[1]);
            record->arg_count = 1;
        }
        else
        {
            writer->func_idBuilder.Append(func_id);
        }
        writer->categoryBuilder.Append(cat);
        writer->ioCategoryBuilder.Append(io_cat);
        writer->levelBuilder.Append(record->call_depth);
        writer->hostnameBuilder.Append(writer->directory.hostname);
        writer->procnameBuilder.Append(proc_name);
        writer->appBuilder.Append(writer->directory.app);
        writer->procIdBuilder.Append(process_hash);
        writer->fileIdBuilder.Append(file_hash);
        writer->index++;
        writer->indexBuilder.Append(writer->index);
        writer->accessPatternBuilder.Append(access_pattern);
        if (writer->index % writer->NUM_ROWS == 0)
        {
            writer->procIdBuilder.Finish(&writer->procIdArray);
            writer->fileIdBuilder.Finish(&writer->fileIdArray);
            writer->indexBuilder.Finish(&writer->indexArray);
            writer->procBuilder.Finish(&writer->procArray);
            writer->sizeBuilder.Finish(&writer->sizeArray);
            writer->rankBuilder.Finish(&writer->rankArray);
            writer->threadBuilder.Finish(&writer->threadArray);
            writer->categoryBuilder.Finish(&writer->categoryArray);
            writer->ioCategoryBuilder.Finish(&writer->ioCategoryArray);
            writer->tstartBuilder.Finish(&writer->tstartArray);
            writer->tendBuilder.Finish(&writer->tendArray);
            writer->durationBuilder.Finish(&writer->durationArray);
            writer->bandwidthBuilder.Finish(&writer->bandwidthArray);
            writer->tmidBuilder.Finish(&writer->tmidArray);
            writer->func_idBuilder.Finish(&writer->func_idArray);
            writer->levelBuilder.Finish(&writer->levelArray);
            writer->hostnameBuilder.Finish(&writer->hostnameArray);
            writer->filenameBuilder.Finish(&writer->filenameArray);
            writer->appBuilder.Finish(&writer->appArray);
            writer->procnameBuilder.Finish(&writer->procnameArray);
            writer->accessPatternBuilder.Finish(&writer->accessPatternArray);

            auto table = arrow::Table::Make(writer->schema, {writer->indexArray,
                                                             writer->procArray,
                                                             writer->rankArray,
                                                             writer->threadArray,
                                                             writer->categoryArray,
                                                             writer->ioCategoryArray,
                                                             writer->tstartArray,
                                                             writer->tendArray,
                                                             writer->func_idArray,
                                                             writer->levelArray,
                                                             writer->hostnameArray,
                                                             writer->appArray,
                                                             writer->procnameArray,
                                                             writer->filenameArray,
                                                             writer->sizeArray,
                                                             writer->accessPatternArray,
                                                             writer->bandwidthArray,
                                                             writer->durationArray,
                                                             writer->tmidArray,
                                                             writer->fileIdArray,
                                                             writer->procIdArray});

            char path[256];
            sprintf(path, "%s_%d.parquet", writer->base_file, writer->row_group);
            DFANALYZER_LOGINFO("Writing %s on rank %d", path, writer->rank);
            PARQUET_ASSIGN_OR_THROW(auto outfile, arrow::io::FileOutputStream::Open(path));
            PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024 * 1024 * 128));

            writer->row_group++;
            writer->procIdBuilder = arrow::Int64Builder();
            writer->procIdArray.reset();
            writer->fileIdBuilder = arrow::Int64Builder();
            writer->fileIdArray.reset();
            writer->indexBuilder = arrow::Int64Builder();
            writer->indexArray.reset();
            writer->procBuilder = arrow::Int64Builder();
            writer->procArray.reset();
            writer->sizeBuilder = arrow::Int64Builder();
            writer->sizeArray.reset();
            writer->tmidBuilder = arrow::Int64Builder();
            writer->tmidArray.reset();
            writer->rankBuilder = arrow::Int32Builder();
            writer->rankArray.reset();
            writer->threadBuilder = arrow::Int32Builder();
            writer->threadArray.reset();
            writer->categoryBuilder = arrow::Int32Builder();
            writer->categoryArray.reset();
            writer->ioCategoryBuilder = arrow::Int32Builder();
            writer->ioCategoryArray.reset();
            writer->levelBuilder = arrow::Int32Builder();
            writer->levelArray.reset();
            writer->accessPatternBuilder = arrow::Int32Builder();
            writer->accessPatternArray.reset();

            writer->hostnameBuilder = arrow::StringBuilder();
            writer->hostnameArray.reset();
            writer->procnameBuilder = arrow::StringBuilder();
            writer->procnameArray.reset();
            writer->filenameBuilder = arrow::StringBuilder();
            writer->filenameArray.reset();

            writer->tstartBuilder = arrow::FloatBuilder();
            writer->tstartArray.reset();
            writer->tendBuilder = arrow::FloatBuilder();
            writer->tendArray.reset();
            writer->durationBuilder = arrow::FloatBuilder();
            writer->durationArray.reset();
            writer->bandwidthBuilder = arrow::FloatBuilder();
            writer->bandwidthArray.reset();

            writer->func_idBuilder = arrow::StringBuilder();
            writer->func_idArray.reset();
            writer->appBuilder = arrow::StringBuilder();
            writer->appArray.reset();
        }
    }
    catch (...)
    {
        std::exception_ptr p = std::current_exception();
        handle_eptr(p);
    }
}
int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }

int process_rank(char *parquet_file_dir, int rank, Directory dir, ParquetWriter *writer)
{
    writer->directory = dir;
    writer->rank = rank;
    CST *cst = reader_get_cst(&reader, 0);
    if (cst->entries > 0)
    {
        recorder_decode_records(&reader, rank, handle_one_record, writer);
        DFANALYZER_LOGINFO("rank %d finished, unique call signatures: %d", rank, cst->entries);
        return cst->entries;
    }
    return 0;
}
#include <chrono>
class Timer
{
public:
    Timer() : elapsed_time(0) {}
    void resumeTime() { t1 = std::chrono::high_resolution_clock::now(); }
    double pauseTime()
    {
        auto t2 = std::chrono::high_resolution_clock::now();
        elapsed_time += std::chrono::duration<double>(t2 - t1).count();
        return elapsed_time;
    }
    double getElapsedTime() { return elapsed_time; }

private:
    std::chrono::high_resolution_clock::time_point t1;
    double elapsed_time;
};
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    // struct sigaction sa;
    // sa.sa_handler = signal_handler;
    // sigemptyset(&sa.sa_mask);
    // sa.sa_flags = SA_RESTART;
    // sigaction(SIGSEGV, &sa, NULL);
    // sigaction(SIGUSR1, &sa, NULL);
    // sigaction(SIGABRT, &sa, NULL);
    // sigaction(SIGHUP, &sa, NULL);
    // sigaction(SIGTERM, &sa, NULL);
    char *dfa_log_level = getenv("DFANALYZER_LOG_LEVEL");
    if (dfa_log_level == nullptr)
    {
        DFANALYZER_LOGGER->level = cpplogger::LoggerType::LOG_ERROR;
        DFANALYZER_LOGINFO("Enabling ERROR logging", "");
    }
    else
    {
        if (strcmp(dfa_log_level, "INFO") == 0)
        {
            DFANALYZER_LOGGER->level = cpplogger::LoggerType::LOG_INFO;
            DFANALYZER_LOGINFO("Enabling INFO logging", "");
        }
        else if (strcmp(dfa_log_level, "DEBUG") == 0)
        {
            DFANALYZER_LOGINFO("Enabling DEBUG logging", "");
            DFANALYZER_LOGGER->level = cpplogger::LoggerType::LOG_DEBUG;
        }
    }
    char parquet_file_dir[256], parquet_file_path[256];
    sprintf(parquet_file_dir, "%s/_parquet", argv[1]);
    mkdir(parquet_file_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    auto ordered_map = std::map<Directory, std::string>();
    if (mpi_rank == 0)
        mkdir(parquet_file_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    std::string base_path(argv[1]);
    bool steps = false;
    for (const auto &entry : fs::directory_iterator(argv[1]))
    {
        if (fs::is_directory(entry))
        {
            std::string dir_string{entry.path().u8string()};
            const size_t last_slash_idx = dir_string.rfind('/');
            std::string directory_name;
            if (std::string::npos != last_slash_idx)
            {
                directory_name = dir_string.substr(last_slash_idx + 1, std::string::npos - 1);
            }

            auto directory = Directory(directory_name);
            if (directory_name != "_parquet")
            {
                ordered_map.insert({directory, dir_string});
            }
            steps = true;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int step = 0;
    if (ordered_map.size() == 0)
    {
        auto dummy_directory_name = "localhost-user-app1-1-1.0";
        auto directory = Directory(dummy_directory_name);
        ordered_map.insert({directory, argv[1]});
    }
    int num_steps = ordered_map.size();
    int n = max(int(ceil(num_steps / mpi_size)), 1);
    int start_step = n * mpi_rank;
    int end_step = min(num_steps, n * (mpi_rank + 1));
    int completed = 0;
    int prev = 0;
    char parquet_filename_path[256];
    sprintf(parquet_filename_path, "%s/%d", parquet_file_dir, mpi_rank);
    ParquetWriter writer(parquet_filename_path);
    Timer step_timer;
    int mpi_steps = 0;
    int workflow_steps = 0;
    for (auto x : ordered_map)
    {
        // std::cout << "Recorder starts " << x.second.c_str() << std::endl;
        recorder_init_reader(x.second.c_str(), &reader);
        if (reader.metadata.total_ranks > 1)
            mpi_steps += 1;
        else
            workflow_steps += 1;
    }
    if (mpi_rank == 0)
        DFANALYZER_LOGPRINT("Workflow has %d mpi steps and %d workflow steps out of total %d steps", mpi_steps, workflow_steps, workflow_steps + mpi_steps);
    for (step = 0; workflow_steps > 0 && step < n; ++step)
    {
        int map_element = mpi_rank * n + step;
        auto x = ordered_map.begin();
        std::advance(x, map_element);
        step_timer.resumeTime();
        int entries = 0;
        if (x != ordered_map.end())
        {
            recorder_init_reader(x->second.c_str(), &reader);
            if (reader.metadata.total_ranks == 1)
            {
                DFANALYZER_LOGINFO("Converting workflow step %d of %d of rank 0 in %s by rank %d", step + 1, workflow_steps, x->first.directory.c_str(), mpi_rank);
                int entry = process_rank(parquet_file_dir, 0, x->first, &writer);
                if (entry == 0)
                    DFANALYZER_LOGERROR("Incomplete trace for rank %d in directory %s", 0, x->second.c_str())
                entries += entry;
            }
            recorder_free_reader(&reader);
        }
        int total_entries;
        MPI_Reduce(&entries, &total_entries, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        step_timer.pauseTime();
        if (mpi_rank == 0)
            DFANALYZER_LOGPRINT("Processed workflow step %d of %d with %d entries in %f secs", (step + 1) * mpi_size, workflow_steps, total_entries, step_timer.getElapsedTime());
    }
    completed = 0;
    if (mpi_steps > 0)
    {
        for (auto x : ordered_map)
        {
            step_timer.resumeTime();
            recorder_init_reader(x.second.c_str(), &reader);
            if (reader.metadata.total_ranks > 1)
            {
                int entries = 0;
                int n = max(reader.metadata.total_ranks / mpi_size, 1);
                int start_rank = n * mpi_rank;
                int end_rank = min(reader.metadata.total_ranks, n * (mpi_rank + 1));
                for (int rank = start_rank; rank < end_rank; rank++)
                {
                    DFANALYZER_LOGINFO("Converting mpi step %d of %d of rank %d in %s by rank %d", step + 1, num_steps, rank, x.first.directory.c_str(), mpi_rank);
                    int entry = process_rank(parquet_file_dir, rank, x.first, &writer);
                    if (entry == 0)
                        DFANALYZER_LOGERROR("Incomplete trace for rank %d in directory %s", rank, x.second.c_str())
                    entries += entry;
                    auto rank_index = rank - start_rank;
                    if (rank_index == n - 1)
                    {
                        DFANALYZER_LOGINFO("Completed ranks %d of %d by rank %d", rank_index + 1, n, mpi_rank);
                    }
                }
                recorder_free_reader(&reader);
                int total_entries;
                MPI_Reduce(&entries, &total_entries, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
                step_timer.pauseTime();
                completed++;
                if (mpi_rank == 0)
                    DFANALYZER_LOGPRINT("Processed mpi step %d of %d with %d entries in %f secs", completed, mpi_steps, total_entries, step_timer.getElapsedTime());
            }
            step++;
            if (prev != completed)
            {
                prev = completed;
            }
        }
    }
    DFANALYZER_LOGPRINT("Completed %d by rank %d", completed, mpi_rank);
    writer.finish();

    long long int global_proc_id_min, global_proc_id_max;
    long long int global_file_id_min, global_file_id_max;
    long long int global_max_tend;
    long long int max_tend_d = writer.max_tend;
    long long int min_file_d = writer.min_file_id, max_file_d = writer.max_file_id;
    long long int min_proc_d = writer.min_proc_id, max_proc_d = writer.max_proc_id;

    MPI_Reduce(&max_tend_d, &global_max_tend, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&min_proc_d, &global_proc_id_min, 1, MPI_LONG_LONG_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_proc_d, &global_proc_id_max, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&min_file_d, &global_file_id_min, 1, MPI_LONG_LONG_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&max_file_d, &global_file_id_max, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    // unsigned long total_transfer_size;
    // MPI_Reduce(&writer.sum_transfer_size, &total_transfer_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    // double total_bandwidth;
    // MPI_Reduce(&writer.sum_bandwidth, &total_bandwidth, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    // unsigned long total_count;
    // MPI_Reduce(&writer.record_count, &total_count, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    using json = nlohmann::json;

    if (mpi_rank == 0)
    {
        int64_t tmid[2] = {0, global_max_tend};
        int64_t proc_id[2] = {global_proc_id_min, global_proc_id_max};
        int64_t file_id[2] = {global_file_id_min, global_file_id_max};
        DFANALYZER_LOGINFO("file_id %lld %lld", global_file_id_min, global_file_id_max);
        DFANALYZER_LOGINFO("proc_id %lld %lld", global_proc_id_min, global_proc_id_max);
        json j = {
            {"tmid", tmid},
            {"proc_id", proc_id},
            {"file_id", file_id},
        };
        char global_json[256];
        sprintf(global_json, "%s/_parquet/global.json", argv[1]);
        std::ofstream out(global_json);
        out << j;
        out.close();
        DFANALYZER_LOGPRINT("Written Global Json file by rank %d", mpi_rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
