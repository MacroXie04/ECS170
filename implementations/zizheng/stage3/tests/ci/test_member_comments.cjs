/* Execute the actual inline Actions comment script against mocked GitHub APIs.
 * No network, tokens, external packages, training, or archived results are used.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const workflow = fs.readFileSync(path.join(root, '.github/workflows/stage2-member.yml'), 'utf8');
const lines = workflow.split('\n');
const beginning = lines.findIndex(line => line === '          script: |');
assert.notEqual(beginning, -1, 'Missing inline comment script');
const body = [];
for (const line of lines.slice(beginning + 1)) {
  if (line && !line.startsWith('            ')) break;
  body.push(line.slice(12));
}
const program = new vm.Script(`(async () => {\n${body.join('\n')}\n})()`);
const members = ['hongzhe', 'aileen', 'zizheng', 'xiaoyu'];
const expected = {
  hongzhe: {names: ['shallow_adam_ce', 'deep_adam_ce', 'deep_sgd_ce', 'deep_adam_mse'],
    marker: '<!-- ecs170-stage2-ci -->', fullName: 'Hongzhe Xie', epochs: 15, seed: 42, batch: 256, pr: 1},
  aileen: {names: ['tuned_mlp'], marker: '<!-- ecs170-stage2-ci-aileen -->',
    fullName: 'Aileen Zheng', epochs: 25, seed: 2, batch: 128, pr: 2},
  zizheng: {names: ['baseline', 'depth2', 'depth3', 'squared_error', 'sgd', 'dropout', 'weight_decay', 'final'],
    marker: '<!-- ecs170-stage2-ci-zizheng -->', fullName: 'Zizheng Wang', epochs: 12, seed: 42, batch: 128, pr: 3},
  xiaoyu: {names: ['baseline', 'deeper', 'sgd_lr_001', 'mse_loss', 'lower_lr'],
    marker: '<!-- ecs170-stage2-ci-xiaoyu -->', fullName: 'Xiaoyu Feng', epochs: 10, seed: 2, batch: 256, pr: 4}
};

function reportFor(member) {
  const spec = expected[member];
  return {implementation: member, train_count: 60000, test_count: 10000,
    environment: {device: 'cpu', python: '3.13.12'},
    rows: spec.names.map(name => ({name, split: member === 'zizheng' && name !== 'final' ? 'validation' : 'test',
      accuracy: .987, macro_f1: .986, precision_macro: .985, recall_macro: .984,
      epochs: spec.epochs, seed: spec.seed, batch_size: spec.batch}))};
}

async function execute(member, options = {}) {
  const comments = options.comments || [];
  const calls = {created: [], updated: [], summaries: [], info: []};
  const github = {
    rest: {
      pulls: {get: async () => ({data: {state: options.state || 'open', head: {sha: options.head || 'current-head'}}})},
      issues: {
        listComments: Symbol('listComments'),
        createComment: async data => {
          calls.created.push(data);
          comments.push({id: 1000 + comments.length, user: {login: 'github-actions[bot]'}, body: data.body});
        },
        updateComment: async data => {
          calls.updated.push(data);
          comments.find(comment => comment.id === data.comment_id).body = data.body;
        }
      }
    },
    paginate: async (method, arguments_) => {
      assert.equal(method, github.rest.issues.listComments);
      assert.equal(arguments_.issue_number, 5);
      return comments;
    }
  };
  const summary = {addRaw(text) {calls.summaries.push(text); return this;}, async write() {}};
  const environment = {MEMBER: member, TRAIN_STATUS: options.status || 'success',
    REPORT_JSON: options.json === undefined ? JSON.stringify(options.report || reportFor(member)) : options.json};
  await program.runInNewContext({github,
    context: {payload: {pull_request: {number: 5, head: {sha: 'current-head'}}},
      repo: {owner: 'MacroXie04', repo: 'ECS170'}, serverUrl: 'https://github.com', runId: 123},
    core: {summary, info: text => calls.info.push(text)}, process: {env: environment}
  }, {timeout: 1000});
  return {comments, calls};
}

async function main() {
  let assertions = 0;
  const provenance = JSON.parse(fs.readFileSync(path.join(root, 'provenance.json'), 'utf8'));
  for (const member of members) {
    const original = provenance.implementations.find(item => item.id === member);
    assert.equal(original.author, expected[member].fullName);
    assert.equal(original.pull_request, expected[member].pr);
  }
  assertions += 8;

  // Jobs can finish independently; their four markers must never overwrite one another.
  const comments = [];
  for (const member of members) {
    const {calls} = await execute(member, {comments});
    assert.equal(calls.created.length, 1);
    assert.equal(calls.updated.length, 0);
    const posted = calls.created[0].body;
    assert.ok(posted.startsWith(expected[member].marker));
    assert.ok(posted.includes(expected[member].fullName));
    assert.ok(posted.includes(`/pull/${expected[member].pr}`));
    assert.ok(posted.includes('current-head'));
    assert.ok(posted.includes('98.70%'));
    assert.equal(calls.summaries.length, 1);
    if (member === 'zizheng') {
      assert.equal((posted.match(/\| validation \|/g) || []).length, 7);
      assert.equal((posted.match(/\| test \|/g) || []).length, 1);
    }
    assertions += 8;
  }
  assert.equal(comments.length, 4);
  assert.equal(new Set(comments.map(comment => comment.body.split('\n')[0])).size, 4);
  assertions += 2;

  // A rerun updates only the matching bot comment, including the old Hongzhe marker.
  for (const member of members) {
    const before = comments.map(comment => ({...comment}));
    const {calls} = await execute(member, {comments, report: {...reportFor(member),
      rows: reportFor(member).rows.map(row => ({...row, accuracy: .98}))}});
    assert.equal(calls.updated.length, 1);
    assert.equal(calls.created.length, 0);
    assert.equal(calls.updated[0].comment_id, before.find(comment => comment.body.startsWith(expected[member].marker)).id);
    for (const comment of before.filter(comment => !comment.body.startsWith(expected[member].marker))) {
      assert.equal(comments.find(current => current.id === comment.id).body, comment.body);
    }
    assertions += 6;
  }

  // User comments containing a marker are never edited by the automation.
  for (const member of members) {
    const human = {id: 1, user: {login: 'human'}, body: expected[member].marker + '\nhuman note'};
    const {calls} = await execute(member, {comments: [human]});
    assert.equal(calls.created.length, 1);
    assert.equal(calls.updated.length, 0);
    assert.equal(human.body, expected[member].marker + '\nhuman note');
    assertions += 3;
  }

  // Failure removes outdated accuracy for only this member and never parses its old payload.
  for (const member of members) {
    const {calls} = await execute(member, {comments, status: 'failure', json: 'not json'});
    assert.equal(calls.updated.length, 1);
    assert.ok(calls.updated[0].body.includes('No accuracy is reported'));
    assert.ok(!calls.updated[0].body.includes('| Model |'));
    assert.ok(!calls.updated[0].body.includes('%'));
    assertions += 4;
  }

  for (const member of members) {
    for (const options of [{head: 'new-head'}, {state: 'closed'}]) {
      const {calls} = await execute(member, options);
      assert.equal(calls.created.length + calls.updated.length + calls.summaries.length, 0);
      assertions += 1;
    }
  }

  // Successful jobs still cannot publish malformed, cross-member, or wrong-protocol metrics.
  for (const member of members) {
    const mutations = [
      report => {report.implementation = 'someone-else';},
      report => {report.train_count = 200;},
      report => {report.test_count = 40;},
      report => {report.environment = null;},
      report => {report.rows.pop();},
      report => {report.rows.push({...report.rows[0]});},
      report => {report.rows[0].name = 'unknown-model';},
      report => {report.rows[0].split = 'training';},
      report => {report.rows[0].accuracy = -1;},
      report => {report.rows[0].macro_f1 = 2;},
      report => {report.rows[0].precision_macro = null;},
      report => {report.rows[0].recall_macro = '0.98';},
      report => {report.rows[0].epochs = 0;},
      report => {report.rows[0].epochs = 26;},
      report => {report.rows[0].epochs = 1.5;},
      report => {report.rows[0].seed = 999;},
      report => {report.rows[0].batch_size = 1;}
    ];
    if (member !== 'zizheng') mutations.push(report => {report.rows[0].epochs = 2;});
    for (const mutate of mutations) {
      const report = reportFor(member);
      mutate(report);
      const unchanged = JSON.stringify(comments);
      await assert.rejects(execute(member, {report, comments}), /Invalid/);
      assert.equal(JSON.stringify(comments), unchanged);
      assertions += 2;
    }
    await assert.rejects(execute(member, {json: 'not-json'}));
    assertions += 1;
  }
  await assert.rejects(execute('unknown', {json: '{}'}), /Unknown member/);
  assertions += 1;
  console.log(`PASS: ${assertions} member-comment assertions; four independent comments, no network calls.`);
}

main().catch(error => {console.error(error); process.exitCode = 1;});
